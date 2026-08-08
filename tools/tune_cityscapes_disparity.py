#!/usr/bin/env python3
"""Grid-search ``cluster_disparity`` parameters on Cityscapes instance GT.

Clusters are treated as class-agnostic mask proposals.  For every ground-truth
thing instance, the script finds the cluster with the highest IoU and reports:

* mean best IoU;
* oracle recall at IoU 0.50 and 0.75;
* cluster false-positive rate and precision;
* object split/fragmentation rate;
* average number of clusters per image;
* clustering runtime.

The default ranking score is the mean of recall@0.50 and recall@0.75.  An
optional query-budget penalty can discourage parameter sets that create too
many transformer queries.  Optional false-positive and split penalties can
also make these failure modes influence the ranking.
"""

import argparse
import csv
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth_clustering_torch import cluster_disparity


CITYSCAPES_THING_IDS = {24, 25, 26, 27, 28, 31, 32, 33}


def comma_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def comma_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def detectron_resize_shape(height, width, short_edge, max_size):
    """Match Detectron2 ResizeShortestEdge output-size calculation."""
    scale = float(short_edge) / min(height, width)
    new_height = height * scale
    new_width = width * scale
    if max(new_height, new_width) > max_size:
        scale = float(max_size) / max(new_height, new_width)
        new_height *= scale
        new_width *= scale
    return int(new_height + 0.5), int(new_width + 0.5)


def add_training_augmentation_plans(samples, args):
    """Create deterministic resize/crop draws shared by every grid setting."""
    if not args.training_augmentations:
        return [(*sample, None) for sample in samples]

    rng = random.Random(args.seed)
    planned_samples = []
    # Cityscapes disparity and instance maps use the native 1024x2048 size.
    native_height, native_width = 1024, 2048
    for sample in samples:
        for _ in range(args.augmentation_repeats):
            short_edge = rng.choice(args.train_min_sizes)
            new_height, new_width = detectron_resize_shape(
                native_height,
                native_width,
                short_edge,
                args.train_max_size,
            )
            crop_height = min(args.crop_height, new_height)
            crop_width = min(args.crop_width, new_width)
            max_y = new_height - crop_height
            max_x = new_width - crop_width
            crop_y = rng.randint(0, max_y) if max_y > 0 else 0
            crop_x = rng.randint(0, max_x) if max_x > 0 else 0
            plan = {
                "height": new_height,
                "width": new_width,
                "area_scale": (
                    (new_height / native_height)
                    * (new_width / native_width)
                ),
                "crop_y": crop_y,
                "crop_x": crop_x,
                "crop_height": crop_height,
                "crop_width": crop_width,
            }
            planned_samples.append((*sample, plan))
    return planned_samples


def discover_samples(root, split):
    disparity_root = root / "disparity" / split
    samples = []
    for disparity in sorted(disparity_root.glob("*/*_disparity.png")):
        city = disparity.parent.name
        prefix = disparity.name[: -len("_disparity.png")]
        camera = root / "camera" / split / city / f"{prefix}_camera.json"
        gt = root / "gtFine" / split / city / f"{prefix}_gtFine_instanceIds.png"
        if not camera.is_file():
            raise FileNotFoundError(camera)
        if not gt.is_file():
            raise FileNotFoundError(gt)
        samples.append((disparity, camera, gt))
    if not samples:
        raise FileNotFoundError(f"No disparity images found below {disparity_root}")
    return samples


def nested_number(data, alternatives, path):
    for keys in alternatives:
        current = data
        try:
            for key in keys:
                current = current[key]
            return float(current)
        except (KeyError, TypeError):
            continue
    raise KeyError(f"Could not find {path} in camera JSON")


def load_camera(path):
    with path.open("r") as handle:
        data = json.load(handle)
    fx = nested_number(data, (("intrinsic", "fx"), ("fx",)), "fx")
    fy = nested_number(data, (("intrinsic", "fy"), ("fy",)), "fy")
    cx = nested_number(
        data, (("intrinsic", "u0"), ("intrinsic", "cx"), ("cx",)), "cx/u0"
    )
    cy = nested_number(
        data, (("intrinsic", "v0"), ("intrinsic", "cy"), ("cy",)), "cy/v0"
    )
    baseline = nested_number(
        data, (("extrinsic", "baseline"), ("baseline",)), "baseline"
    )
    return fx, fy, cx, cy, baseline


def decode_disparity(path):
    raw = np.asarray(Image.open(path), dtype=np.float32)
    disparity = np.zeros_like(raw, dtype=np.float32)
    valid = raw > 0
    disparity[valid] = (raw[valid] - 1.0) / 256.0
    return torch.from_numpy(disparity)


def load_instance_ids(path):
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.int64).copy())


def apply_training_augmentation(disparity, instance_ids, camera, plan):
    """Apply planned ResizeShortestEdge and crop to one sample."""
    if plan is None:
        return disparity, instance_ids, camera

    old_height, old_width = disparity.shape
    new_height = plan["height"]
    new_width = plan["width"]
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    disparity = F.interpolate(
        disparity[None, None],
        size=(new_height, new_width),
        mode="nearest",
    )[0, 0]
    disparity = disparity * scale_x
    instance_ids = F.interpolate(
        instance_ids[None, None].float(),
        size=(new_height, new_width),
        mode="nearest",
    )[0, 0].long()

    camera = camera.clone()
    camera[0] *= scale_x
    camera[1] *= scale_y
    camera[2] *= scale_x
    camera[3] *= scale_y

    y0 = plan["crop_y"]
    x0 = plan["crop_x"]
    crop_height = plan["crop_height"]
    crop_width = plan["crop_width"]
    disparity = disparity[y0 : y0 + crop_height, x0 : x0 + crop_width]
    instance_ids = instance_ids[
        y0 : y0 + crop_height, x0 : x0 + crop_width
    ]
    camera[2] -= x0
    camera[3] -= y0
    return disparity, instance_ids, camera


def resize_batch(disparity, instance_ids, camera, scale):
    if scale == 1.0:
        return disparity, instance_ids, camera
    height, width = disparity.shape[-2:]
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    scale_x = new_width / width
    scale_y = new_height / height
    disparity = F.interpolate(
        disparity[:, None], size=(new_height, new_width), mode="nearest"
    )[:, 0]
    disparity = disparity * scale_x
    instance_ids = F.interpolate(
        instance_ids[:, None].float(),
        size=(new_height, new_width),
        mode="nearest",
    )[:, 0].long()
    camera = camera.clone()
    camera[:, 0] *= scale_x
    camera[:, 1] *= scale_y
    camera[:, 2] *= scale_x
    camera[:, 3] *= scale_y
    return disparity, instance_ids, camera


def local_gt_map(instance_ids):
    """Return (H,W) values 0=background and 1..G=thing instances."""
    flat = instance_ids.reshape(-1)
    unique_ids = torch.unique(flat)
    keep = [
        int(instance_id)
        for instance_id in unique_ids.tolist()
        if instance_id >= 1000
        and instance_id // 1000 in CITYSCAPES_THING_IDS
    ]
    result = torch.zeros_like(instance_ids, dtype=torch.long)
    for local_id, instance_id in enumerate(keep, start=1):
        result[instance_ids == instance_id] = local_id
    return result, len(keep)


def one_to_one_matches(iou, threshold):
    """Greedily match clusters and GT instances one-to-one above an IoU."""
    candidates = torch.nonzero(iou >= threshold, as_tuple=False)
    if candidates.numel() == 0:
        return 0, 0.0

    candidate_ious = iou[candidates[:, 0], candidates[:, 1]]
    order = torch.argsort(candidate_ious, descending=True)
    sorted_candidates = candidates[order].detach().cpu().tolist()
    sorted_ious = candidate_ious[order].detach().cpu().tolist()
    used_gt = set()
    used_cluster = set()
    matches = 0
    matched_iou_sum = 0.0

    for (gt_index, cluster_index), match_iou in zip(
        sorted_candidates, sorted_ious
    ):
        if gt_index in used_gt or cluster_index in used_cluster:
            continue
        used_gt.add(gt_index)
        used_cluster.add(cluster_index)
        matches += 1
        matched_iou_sum += match_iou

    return matches, matched_iou_sum


def image_proposal_metrics(
    labels,
    instance_ids,
    false_positive_purity,
    fragment_min_gt_fraction,
):
    """Measure GT recall, cluster false positives, and GT fragmentation.

    A cluster is a false positive when less than ``false_positive_purity`` of
    its pixels belong to any one thing instance.  Each non-false-positive
    cluster is assigned to the GT instance with which it has the largest
    intersection.  It counts as a fragment only if it also covers at least
    ``fragment_min_gt_fraction`` of that GT instance.
    """
    gt, gt_count = local_gt_map(instance_ids)

    cluster_ids = torch.unique(labels[labels >= 0])
    cluster_count = int(cluster_ids.numel())
    if cluster_count == 0:
        return {
            "best_ious": torch.zeros(gt_count, device=labels.device),
            "cluster_count": 0,
            "false_positive_clusters": 0,
            "represented_gt": 0,
            "split_gt": 0,
            "fragment_count": 0,
            "excess_fragments": 0,
            "matches_50": 0,
            "matched_iou_sum_50": 0.0,
            "matches_75": 0,
            "matched_iou_sum_75": 0.0,
        }

    # searchsorted maps arbitrary/global cluster ids into local 0..K-1 ids.
    valid_cluster = labels >= 0
    local_cluster = torch.searchsorted(cluster_ids, labels[valid_cluster])
    cluster_area = torch.bincount(local_cluster, minlength=cluster_count).float()

    if gt_count == 0:
        return {
            "best_ious": torch.empty(0, device=labels.device),
            "cluster_count": cluster_count,
            "false_positive_clusters": cluster_count,
            "represented_gt": 0,
            "split_gt": 0,
            "fragment_count": 0,
            "excess_fragments": 0,
            "matches_50": 0,
            "matched_iou_sum_50": 0.0,
            "matches_75": 0,
            "matched_iou_sum_75": 0.0,
        }

    joint = valid_cluster & (gt > 0)
    joint_cluster = torch.searchsorted(cluster_ids, labels[joint])
    joint_gt = gt[joint] - 1
    encoded = joint_gt * cluster_count + joint_cluster
    intersection = torch.bincount(
        encoded, minlength=gt_count * cluster_count
    ).reshape(gt_count, cluster_count).float()

    gt_area = torch.bincount(
        gt[gt > 0] - 1, minlength=gt_count
    ).float()
    union = gt_area[:, None] + cluster_area[None, :] - intersection
    iou = intersection / union.clamp_min(1.0)
    best_ious = iou.max(dim=1).values
    matches_50, matched_iou_sum_50 = one_to_one_matches(iou, 0.50)
    matches_75, matched_iou_sum_75 = one_to_one_matches(iou, 0.75)

    # Assign each cluster to the single GT instance with the greatest pixel
    # intersection. Purity answers: "how much of this cluster is object?"
    best_intersection, assigned_gt = intersection.max(dim=0)
    purity = best_intersection / cluster_area.clamp_min(1.0)
    false_positive = purity < false_positive_purity

    # Ignore tiny accidental contacts when deciding whether an object has been
    # split. A valid fragment must be sufficiently pure and cover a minimum
    # fraction of its assigned GT object.
    assigned_gt_area = gt_area[assigned_gt]
    gt_fraction = best_intersection / assigned_gt_area.clamp_min(1.0)
    is_fragment = (~false_positive) & (
        gt_fraction >= fragment_min_gt_fraction
    )
    fragments_per_gt = torch.bincount(
        assigned_gt[is_fragment], minlength=gt_count
    )
    represented = fragments_per_gt > 0
    split = fragments_per_gt > 1

    return {
        "best_ious": best_ious,
        "cluster_count": cluster_count,
        "false_positive_clusters": int(false_positive.sum()),
        "represented_gt": int(represented.sum()),
        "split_gt": int(split.sum()),
        "fragment_count": int(fragments_per_gt.sum()),
        "excess_fragments": int(
            (fragments_per_gt - 1).clamp_min(0).sum()
        ),
        "matches_50": matches_50,
        "matched_iou_sum_50": matched_iou_sum_50,
        "matches_75": matches_75,
        "matched_iou_sum_75": matched_iou_sum_75,
    }


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.inference_mode()
def evaluate(samples, params, args):
    device = torch.device(args.device)
    iou_sum = 0.0
    gt_count = 0
    recall50 = 0
    recall75 = 0
    cluster_count = 0
    false_positive_clusters = 0
    represented_gt = 0
    split_gt = 0
    fragment_count = 0
    excess_fragments = 0
    matches_50 = 0
    matched_iou_sum_50 = 0.0
    matches_75 = 0
    matched_iou_sum_75 = 0.0
    image_count = 0
    elapsed = 0.0
    effective_min_size_sum = 0.0
    effective_min_size_count = 0

    for sample_batch in batches(samples, args.batch_size):
        augmented_batch = []
        for sample in sample_batch:
            disparity = decode_disparity(sample[0])
            instance_id = load_instance_ids(sample[2])
            sample_camera = torch.tensor(
                load_camera(sample[1]), dtype=torch.float32
            )
            disparity, instance_id, sample_camera = (
                apply_training_augmentation(
                    disparity,
                    instance_id,
                    sample_camera,
                    sample[3],
                )
            )
            augmented_batch.append(
                (disparity, instance_id, sample_camera)
            )

        disparities = torch.stack(
            [item[0] for item in augmented_batch]
        ).to(device, non_blocking=True)
        instance_ids = torch.stack(
            [item[1] for item in augmented_batch]
        ).to(device, non_blocking=True)
        camera = torch.stack(
            [item[2] for item in augmented_batch]
        ).to(device, non_blocking=True)
        disparities, instance_ids, camera = resize_batch(
            disparities, instance_ids, camera, args.scale
        )

        effective_min_sizes = []
        for sample in sample_batch:
            area_scale = (
                sample[3]["area_scale"]
                if sample[3] is not None
                else 1.0
            )
            effective_min_size = params["min_size"]
            if args.adapt_min_size:
                effective_min_size = max(
                    1,
                    round(params["min_size"] * area_scale),
                )
            effective_min_sizes.append(effective_min_size)
        effective_min_size_sum += sum(effective_min_sizes)
        effective_min_size_count += len(effective_min_sizes)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if args.adapt_min_size:
            # The extension accepts a scalar min_size, so retain batching by
            # grouping samples that have the same effective integer threshold.
            grouped_indices = {}
            for batch_index, effective_min_size in enumerate(
                effective_min_sizes
            ):
                grouped_indices.setdefault(
                    effective_min_size, []
                ).append(batch_index)

            label_items = [None] * len(sample_batch)
            for effective_min_size, indices in grouped_indices.items():
                index_tensor = torch.tensor(
                    indices, dtype=torch.long, device=device
                )
                group_camera = camera.index_select(0, index_tensor)
                group_labels = cluster_disparity(
                    disparities.index_select(0, index_tensor),
                    fx=group_camera[:, 0],
                    fy=group_camera[:, 1],
                    cx=group_camera[:, 2],
                    cy=group_camera[:, 3],
                    baseline=group_camera[:, 4],
                    theta_deg=params["theta_deg"],
                    min_size=effective_min_size,
                    ground=params["ground"],
                    ground_thresh_deg=params["ground_thresh_deg"],
                )
                if group_labels.ndim == 2:
                    group_labels = group_labels.unsqueeze(0)
                for group_index, batch_index in enumerate(indices):
                    label_items[batch_index] = group_labels[group_index]
            labels = torch.stack(label_items)
        else:
            labels = cluster_disparity(
                disparities,
                fx=camera[:, 0],
                fy=camera[:, 1],
                cx=camera[:, 2],
                cy=camera[:, 3],
                baseline=camera[:, 4],
                theta_deg=params["theta_deg"],
                min_size=params["min_size"],
                ground=params["ground"],
                ground_thresh_deg=params["ground_thresh_deg"],
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start

        for batch_index in range(labels.shape[0]):
            metrics = image_proposal_metrics(
                labels[batch_index],
                instance_ids[batch_index],
                args.false_positive_purity,
                args.fragment_min_gt_fraction,
            )
            best = metrics["best_ious"]
            iou_sum += float(best.sum())
            gt_count += int(best.numel())
            recall50 += int((best >= 0.50).sum())
            recall75 += int((best >= 0.75).sum())
            cluster_count += metrics["cluster_count"]
            false_positive_clusters += metrics["false_positive_clusters"]
            represented_gt += metrics["represented_gt"]
            split_gt += metrics["split_gt"]
            fragment_count += metrics["fragment_count"]
            excess_fragments += metrics["excess_fragments"]
            matches_50 += metrics["matches_50"]
            matched_iou_sum_50 += metrics["matched_iou_sum_50"]
            matches_75 += metrics["matches_75"]
            matched_iou_sum_75 += metrics["matched_iou_sum_75"]
            image_count += 1

    mean_iou = iou_sum / max(gt_count, 1)
    r50 = recall50 / max(gt_count, 1)
    r75 = recall75 / max(gt_count, 1)
    clusters_per_image = cluster_count / max(image_count, 1)
    false_positive_rate = false_positive_clusters / max(cluster_count, 1)
    cluster_precision = 1.0 - false_positive_rate
    false_positives_per_image = (
        false_positive_clusters / max(image_count, 1)
    )
    split_rate = split_gt / max(represented_gt, 1)
    mean_fragments_per_represented_gt = (
        fragment_count / max(represented_gt, 1)
    )
    excess_fragments_per_gt = excess_fragments / max(gt_count, 1)

    # One-to-one proposal detection statistics. An unmatched cluster is an FP
    # and an unmatched GT instance is an FN. Duplicate fragments therefore
    # become FPs instead of receiving credit for the same object.
    proposal_fp_50 = cluster_count - matches_50
    proposal_fn_50 = gt_count - matches_50
    proposal_precision_50 = matches_50 / max(
        matches_50 + proposal_fp_50, 1
    )
    proposal_recall_50 = matches_50 / max(
        matches_50 + proposal_fn_50, 1
    )
    proposal_f1_50 = (
        2.0
        * proposal_precision_50
        * proposal_recall_50
        / max(proposal_precision_50 + proposal_recall_50, 1e-12)
    )
    proposal_pq_50 = matched_iou_sum_50 / max(
        matches_50 + 0.5 * proposal_fp_50 + 0.5 * proposal_fn_50,
        1e-12,
    )

    proposal_fp_75 = cluster_count - matches_75
    proposal_fn_75 = gt_count - matches_75
    proposal_precision_75 = matches_75 / max(
        matches_75 + proposal_fp_75, 1
    )
    proposal_recall_75 = matches_75 / max(
        matches_75 + proposal_fn_75, 1
    )
    proposal_f1_75 = (
        2.0
        * proposal_precision_75
        * proposal_recall_75
        / max(proposal_precision_75 + proposal_recall_75, 1e-12)
    )
    proposal_pq_75 = matched_iou_sum_75 / max(
        matches_75 + 0.5 * proposal_fp_75 + 0.5 * proposal_fn_75,
        1e-12,
    )

    excess_queries = max(
        clusters_per_image / max(args.query_budget, 1) - 1.0, 0.0
    )
    recall_score = 0.5 * (r50 + r75)
    proposal_f1_mean = 0.5 * (proposal_f1_50 + proposal_f1_75)
    proposal_pq_mean = 0.5 * (proposal_pq_50 + proposal_pq_75)
    ranking_score = {
        "recall": recall_score,
        "f1": proposal_f1_mean,
        "pq": proposal_pq_mean,
    }[args.ranking_metric]
    score = (
        ranking_score
        - args.cluster_penalty * excess_queries
        - args.false_positive_penalty * false_positive_rate
        - args.split_penalty * split_rate
    )
    return {
        **params,
        "score": score,
        "ranking_metric": args.ranking_metric,
        "recall_score": recall_score,
        "mean_best_iou": mean_iou,
        "recall_50": r50,
        "recall_75": r75,
        "cluster_precision": cluster_precision,
        "false_positive_rate": false_positive_rate,
        "false_positive_clusters_per_image": false_positives_per_image,
        "split_rate_represented_gt": split_rate,
        "mean_fragments_per_represented_gt": (
            mean_fragments_per_represented_gt
        ),
        "excess_fragments_per_gt": excess_fragments_per_gt,
        "proposal_tp_50": matches_50,
        "proposal_fp_50": proposal_fp_50,
        "proposal_fn_50": proposal_fn_50,
        "proposal_precision_50": proposal_precision_50,
        "proposal_recall_50": proposal_recall_50,
        "proposal_f1_50": proposal_f1_50,
        "proposal_pq_50": proposal_pq_50,
        "proposal_tp_75": matches_75,
        "proposal_fp_75": proposal_fp_75,
        "proposal_fn_75": proposal_fn_75,
        "proposal_precision_75": proposal_precision_75,
        "proposal_recall_75": proposal_recall_75,
        "proposal_f1_75": proposal_f1_75,
        "proposal_pq_75": proposal_pq_75,
        "proposal_f1_mean": proposal_f1_mean,
        "proposal_pq_mean": proposal_pq_mean,
        "adapt_min_size": args.adapt_min_size,
        "mean_effective_min_size": (
            effective_min_size_sum / max(effective_min_size_count, 1)
        ),
        "clusters_per_image": clusters_per_image,
        "gt_instances": gt_count,
        "seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / max(image_count, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, default=Path("cluster_search.csv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-augmentations",
        action="store_true",
        help=(
            "Apply deterministic draws from training ResizeShortestEdge and "
            "random crop before the model clustering scale."
        ),
    )
    parser.add_argument(
        "--train-min-sizes",
        type=comma_ints,
        default=[
            512, 614, 716, 819, 921, 1024, 1126, 1228,
            1331, 1433, 1536, 1638, 1740, 1843, 1945, 2048,
        ],
    )
    parser.add_argument("--train-max-size", type=int, default=4096)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument(
        "--adapt-min-size",
        action="store_true",
        help=(
            "Interpret each grid min_size as a base value and multiply it by "
            "the mapper resize area scale (scale_x * scale_y) per image."
        ),
    )
    parser.add_argument(
        "--augmentation-repeats",
        type=int,
        default=1,
        help="Number of fixed random augmentation draws per source image.",
    )
    parser.add_argument("--theta-deg", type=comma_floats, default=[3, 4, 5, 6, 7])
    parser.add_argument("--min-size", type=comma_ints, default=[100, 200, 400, 800])
    parser.add_argument(
        "--ground", choices=("false", "true", "both"), default="both"
    )
    parser.add_argument(
        "--ground-thresh-deg", type=comma_floats, default=[3, 5, 10, 15]
    )
    parser.add_argument("--query-budget", type=int, default=100)
    parser.add_argument("--cluster-penalty", type=float, default=0.0)
    parser.add_argument(
        "--ranking-metric",
        choices=("recall", "f1", "pq"),
        default="recall",
        help=(
            "Primary score before optional penalties: legacy oracle recall, "
            "one-to-one proposal F1, or class-agnostic proposal PQ."
        ),
    )
    parser.add_argument(
        "--false-positive-purity",
        type=float,
        default=0.5,
        help=(
            "A cluster is a false positive if less than this fraction of its "
            "pixels belongs to any single GT thing instance."
        ),
    )
    parser.add_argument(
        "--fragment-min-gt-fraction",
        type=float,
        default=0.05,
        help=(
            "A cluster must cover at least this fraction of its assigned GT "
            "instance to count as an object fragment."
        ),
    )
    parser.add_argument(
        "--false-positive-penalty",
        type=float,
        default=0.0,
        help="Score penalty weight for the cluster false-positive rate.",
    )
    parser.add_argument(
        "--split-penalty",
        type=float,
        default=0.0,
        help="Score penalty weight for the represented-GT split rate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0,1]")
    if args.augmentation_repeats < 1:
        raise ValueError("--augmentation-repeats must be at least 1")
    if not args.train_min_sizes:
        raise ValueError("--train-min-sizes cannot be empty")
    if not 0 <= args.false_positive_purity <= 1:
        raise ValueError("--false-positive-purity must be in [0,1]")
    if not 0 <= args.fragment_min_gt_fraction <= 1:
        raise ValueError("--fragment-min-gt-fraction must be in [0,1]")
    samples = discover_samples(args.cityscapes_root, args.split)
    random.Random(args.seed).shuffle(samples)
    if args.max_images > 0:
        samples = samples[: args.max_images]
    source_image_count = len(samples)
    samples = add_training_augmentation_plans(samples, args)

    ground_values = {
        "false": [False],
        "true": [True],
        "both": [False, True],
    }[args.ground]
    combinations = list(
        itertools.product(
            args.theta_deg,
            args.min_size,
            ground_values,
            args.ground_thresh_deg,
        )
    )
    # The threshold is irrelevant when ground removal is disabled.
    combinations = list(
        dict.fromkeys(
            (theta, size, ground, threshold if ground else 0.0)
            for theta, size, ground, threshold in combinations
        )
    )

    print(
        f"Evaluating {len(combinations)} settings on {source_image_count} "
        f"{args.split} images ({len(samples)} augmentation draws) at model "
        f"clustering scale {args.scale}. Training augmentations: "
        f"{args.training_augmentations}."
    )
    rows = []
    fieldnames = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (theta, size, ground, ground_threshold) in enumerate(
        combinations, start=1
    ):
        params = {
            "theta_deg": theta,
            "min_size": size,
            "ground": ground,
            "ground_thresh_deg": ground_threshold,
        }
        row = evaluate(samples, params, args)
        rows.append(row)
        fieldnames = list(row)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                sorted(rows, key=lambda item: item["score"], reverse=True)
            )
        print(
            f"[{index:>3}/{len(combinations)}] {params} "
            f"score={row['score']:.4f} mIoU={row['mean_best_iou']:.4f} "
            f"R50={row['recall_50']:.4f} R75={row['recall_75']:.4f} "
            f"precision={row['cluster_precision']:.4f} "
            f"split={row['split_rate_represented_gt']:.4f} "
            f"F1={row['proposal_f1_mean']:.4f} "
            f"PQ={row['proposal_pq_mean']:.4f} "
            f"K/img={row['clusters_per_image']:.1f} "
            f"ms/img={row['milliseconds_per_image']:.1f}"
        )

    best = max(rows, key=lambda item: item["score"])
    print("\nBest setting:")
    print(json.dumps(best, indent=2))
    print(f"All results: {args.output}")


if __name__ == "__main__":
    main()
