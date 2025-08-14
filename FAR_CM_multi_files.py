import json
import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Iterable, Tuple


# ----------------------------
# IoU & association (unchanged)
# ----------------------------
def calculate_iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea


def associate_gt_pred(gt_faces, pred_faces, iou_threshold=0.3, score_threshold=0.0, unknown_label="unknown"):
    matches = []
    used_preds = set()
    # For each ground truth face, find best matching prediction
    for gt_idx, gt in enumerate(gt_faces):
        gt_bbox = {
            "x1": gt["bounding_box"]["top_left"]["x"],
            "y1": gt["bounding_box"]["top_left"]["y"],
            "x2": gt["bounding_box"]["bottom_right"]["x"],
            "y2": gt["bounding_box"]["bottom_right"]["y"]
        }
        best_iou = 0
        best_pred_idx = None
        for pred_idx, pred in enumerate(pred_faces):
            if pred_idx in used_preds:
                continue
            pred_bbox = {
                "x1": pred["bbox"]["x1"],
                "y1": pred["bbox"]["y1"],
                "x2": pred["bbox"]["x2"],
                "y2": pred["bbox"]["y2"]
            }
            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            used_preds.add(best_pred_idx)
            pred = pred_faces[best_pred_idx]
            # Apply score threshold: if below, treat as "unknown"
            pred_label = pred["label"]
            if pred.get("score", 1.0) < score_threshold:
                pred_label = unknown_label
            matches.append((gt["name"], pred_label))
        else:
            # No detection matched this GT face: treat as missed ("unknown")
            matches.append((gt["name"], unknown_label))
    return matches


# ----------------------------
# Multi-file loaders + merging
# ----------------------------
def _ensure_iterable_paths(paths_or_glob: Iterable[str]) -> List[Path]:
    """
    Accepts:
      - an iterable of paths/globs
      - or a single string path/glob
    Returns a sorted list of Path objects matched/expanded.
    """
    if isinstance(paths_or_glob, (str, Path)):
        paths_or_glob = [paths_or_glob]
    files: List[Path] = []
    for p in paths_or_glob:
        p = str(p)
        # Expand globs and directories
        if any(ch in p for ch in "*?[]"):
            files.extend(sorted(Path().glob(p)))
        else:
            pth = Path(p)
            if pth.is_dir():
                # include only .json files in a directory
                files.extend(sorted(pth.glob("*.json")))
            else:
                files.append(pth)
    # Final natural-ish sort by name to keep numeric order intact
    def natural_key(s: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s.name)]
    return sorted(list(dict.fromkeys(files)), key=natural_key)


def load_gt_series(gt_files: Iterable[str]) -> Dict[int, List[dict]]:
    """
    Load and merge multiple GT JSON files into a single dict: frame_id -> [faces].
    Each GT file has frame indices starting at 0. We offset each subsequent file
    so frames are globally unique and in sequence.
    """
    files = _ensure_iterable_paths(gt_files)
    gt_by_frame: Dict[int, List[dict]] = {}
    running_offset = 0

    for fp in files:
        with open(fp, "r") as f:
            gt_json = json.load(f)

        # Track max local frame id in this file to compute next offset
        local_max = -1
        for entry in gt_json:
            for face in entry.get("faces", []):
                local_frame = face["frame_id"]  # starts at 0 in every GT file
                global_frame = local_frame + running_offset
                if global_frame not in gt_by_frame:
                    gt_by_frame[global_frame] = []
                # Store a copy with updated frame_id (optional)
                face_copy = dict(face)
                face_copy["frame_id"] = global_frame
                gt_by_frame[global_frame].append(face_copy)
                local_max = max(local_max, local_frame)

        # If file had frames [0..local_max], next file starts after that range
        if local_max >= 0:
            running_offset += (local_max + 1)

    return gt_by_frame


def load_pred_series(pred_files: Iterable[str]) -> Dict[int, List[dict]]:
    """
    Load and merge multiple prediction JSON files into a single dict: frame_id -> [faces].
    Each prediction file starts at "image": "frame0001.png".
    We convert to 0-based indices and then offset by a running total across files.
    """
    files = _ensure_iterable_paths(pred_files)
    pred_by_frame: Dict[int, List[dict]] = {}
    running_offset = 0

    for fp in files:
        with open(fp, "r") as f:
            pred_json = json.load(f)

        local_max = -1
        for entry in pred_json:
            frame_str = entry["image"]  # e.g., "frame0001.png"
            # Extract the numeric portion between 'frame' and optional extension
            # Assumes pattern frame<digits>.<ext>
            num = int(os.path.splitext(os.path.basename(frame_str))[0].replace("frame", ""))
            local_idx = num - 1  # convert to 0-based within this file
            global_idx = local_idx + running_offset
            pred_by_frame[global_idx] = entry.get("faces", [])
            local_max = max(local_max, local_idx)

        if local_max >= 0:
            running_offset += (local_max + 1)

    return pred_by_frame


# --------------------------------------
# Metric computation on merged frame maps
# --------------------------------------
def calculate_far_and_confusion_from_maps(
    gt_by_frame: Dict[int, List[dict]],
    pred_by_frame: Dict[int, List[dict]],
    score_threshold: float = 0.0,
    unknown_label: str = "unknown",
    iou_threshold: float = 0.3
) -> Tuple[float, List[str], List[str]]:
    false_accepts = 0
    valid_attempts = 0
    y_true: List[str] = []
    y_pred: List[str] = []

    for frame_id, gt_faces in gt_by_frame.items():
        pred_faces = pred_by_frame.get(frame_id, [])
        matches = associate_gt_pred(
            gt_faces, pred_faces,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            unknown_label=unknown_label
        )
        for gt_label, pred_label in matches:
            gt_label_lower = gt_label.lower()
            pred_label_lower = pred_label.lower()
            if pred_label_lower != unknown_label.lower():
                valid_attempts += 1
                if pred_label_lower != gt_label_lower:
                    false_accepts += 1
            y_true.append(gt_label_lower)
            y_pred.append(pred_label_lower)

    far = (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0
    return far, y_true, y_pred


# ----------------------------
# Plotting (unchanged)
# ----------------------------
def plot_confusion_matrix(far_value, y_true, y_pred, unknown_label="unknown"):
    df = pd.DataFrame({'Ground Truth': y_true, 'Predicted': y_pred})
    # Remove any accidental unknown in GT (shouldn't be any)
    gt_labels = sorted(set(df["Ground Truth"]) - {"none", "unknown"})
    pred_labels = sorted(set(df["Predicted"]))

    if unknown_label not in pred_labels:
        pred_labels.append(unknown_label)

    # Move unknown to end of pred_labels
    for special in [unknown_label, "none"]:
        if special in pred_labels:
            pred_labels.remove(special)
            pred_labels.append(special)

    confusion_matrix = pd.crosstab(
        df["Ground Truth"], df["Predicted"],
        rownames=["GT"], colnames=["Predicted"],
        dropna=False
    ).reindex(index=gt_labels, columns=pred_labels, fill_value=0)

    correct_predictions = sum(
        confusion_matrix.loc[label, label]
        for label in confusion_matrix.index
        if label in confusion_matrix.columns
    )
    total_predicted = confusion_matrix.sum().sum()
    accuracy_pred = correct_predictions / total_predicted if total_predicted > 0 else 0

    annotations = confusion_matrix.copy().astype(str)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            val = confusion_matrix.iloc[i, j]
            annotations.iloc[i, j] = f"{val}"

    fixed_figsize = (8, 6)
    fig, ax = plt.subplots(figsize=fixed_figsize)
    sns.heatmap(confusion_matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(
        f"GT vs Predicted | Acc: {accuracy_pred:.2%} ({correct_predictions}/{total_predicted})\n"
        f"{' ' * 20}Actual FAR: {far_value:.2f}"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example usage with MANY files
# ----------------------------
if __name__ == "__main__":
    # You can pass:
    #   - a directory (we'll read all *.json inside),
    #   - a glob pattern,
    #   - or a list of explicit file paths.
    #
    # Example 1: directory (all JSONs inside)
    # gt_paths = "ground_truth/5pc/left"  # reads *.json inside this directory
    # pred_paths = "analysis_results/ir101_webface12m/5pc_left"  # reads *.json inside

    # Example 2: glob (sorted naturally)
    # gt_paths = "ground_truth/5pc/left/gt_5pc_vid*_left.json"
    # pred_paths = "analysis_results/ir101_webface12m/5pc_left/recognition_results_vid*.json"

    # Example 3: explicit list
    gt_paths = [
        "ground_truth/5pc/left",
        "ground_truth/5pc/right"
    ]
    pred_paths = [
        "analysis_results/ir101_webface12m/5pc_left",
        "analysis_results/ir101_webface12m/5pc_right"
    ]

    # Load and merge with correct per-file frame offsets
    gt_by_frame = load_gt_series(gt_paths)
    pred_by_frame = load_pred_series(pred_paths)

    # Compute metrics
    score_threshold = 0.415
    far_value, y_true, y_pred = calculate_far_and_confusion_from_maps(
        gt_by_frame, pred_by_frame,
        score_threshold=score_threshold,
        unknown_label="unknown",
        iou_threshold=0.3
    )
    print(f"False Acceptance Rate (FAR) at threshold {score_threshold}: {far_value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(far_value, y_true, y_pred, unknown_label="unknown")
