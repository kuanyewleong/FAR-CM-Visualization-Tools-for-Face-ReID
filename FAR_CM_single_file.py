import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

def calculate_far_and_confusion(gt_json, pred_json, score_threshold=0.0, unknown_label="unknown"):
    false_accepts = 0
    valid_attempts = 0
    y_true = []
    y_pred = []
    gt_by_frame = {}
    for entry in gt_json:
        for face in entry["faces"]:
            frame_id = face["frame_id"]
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append(face)
    # Build a lookup for prediction per frame
    pred_by_frame = {}
    for entry in pred_json:
        frame_str = entry["image"]
        frame_id = int(os.path.splitext(frame_str)[0].replace("frame", "")) - 1
        pred_by_frame[frame_id] = entry["faces"]

    # For each frame with ground truth, process all faces
    for frame_id, gt_faces in gt_by_frame.items():
        pred_faces = pred_by_frame.get(frame_id, [])
        matches = associate_gt_pred(
            gt_faces, pred_faces,
            iou_threshold=0.3,
            score_threshold=score_threshold,
            unknown_label=unknown_label
        )
        for gt_label, pred_label in matches:
            gt_label_lower = gt_label.lower()
            pred_label_lower = pred_label.lower()
            # For FAR: only count if prediction is not "unknown"
            if pred_label_lower != unknown_label.lower():
                valid_attempts += 1
                if pred_label_lower != gt_label_lower:
                    false_accepts += 1
            y_true.append(gt_label_lower)
            y_pred.append(pred_label_lower)
    far = (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0
    return far, y_true, y_pred

def plot_confusion_matrix(far_value, y_true, y_pred, unknown_label="unknown"):
    df = pd.DataFrame({'Ground Truth': y_true, 'Predicted': y_pred})
    # Remove any accidental unknown in GT (shouldn't be any)
    gt_labels = sorted(set(df["Ground Truth"]) - {"none", "unknown"})
    pred_labels = sorted(set(df["Predicted"]))
    
    # Ensure 'unknown' is always included in pred_labels
    # even if it is zero occurrences to maintain figure shape
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
    # total_gt = confusion_matrix.sum(axis=1).sum()
    accuracy_pred = correct_predictions / total_predicted if total_predicted > 0 else 0
    # accuracy_gt = correct_predictions / total_gt if total_gt > 0 else 0

    # Annotate
    annotations = confusion_matrix.copy().astype(str)
    # row_totals = confusion_matrix.sum(axis=1)
    # col_totals = confusion_matrix.sum(axis=0)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            val = confusion_matrix.iloc[i, j]
            # gt_label = confusion_matrix.index[i]
            # pred_label = confusion_matrix.columns[j]
            # row_total = row_totals[gt_label]
            # col_total = col_totals[pred_label]
            annotations.iloc[i, j] = f"{val}" #\n{val}/{row_total} (GT)\n{val}/{col_total} (Pred)"

    
    fixed_figsize = (8, 6)
    fig, ax = plt.subplots(figsize=fixed_figsize)
    # fig, ax = plt.subplots(figsize=(1 + len(pred_labels), 1 + len(gt_labels)))
    sns.heatmap(confusion_matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(
        f"GT vs Predicted | Acc: {accuracy_pred:.2%} ({correct_predictions}/{total_predicted})\n"
        f"{' ' * 20}Actual FAR: {far_value:.2%}"
    )
    # ax.set_title(
    #     f"GT vs Predicted | Acc (Pred): {accuracy_pred:.2%} | Acc (GT): {accuracy_gt:.2%}\n"
    #     f"Correct: {correct_predictions} / Pred: {total_predicted}, GT: {total_gt}"
    # )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    with open("ground_truth/3pc/left/gt_3pc_vid0_left.json", "r") as f:
        gt_json = json.load(f)
    with open("analysis_results/ir101_webface12m/3pc_left/recognition_results_vid0.json", "r") as f:
        pred_json = json.load(f)
    score_threshold = 0.31  # Set your threshold
    far_value, y_true, y_pred = calculate_far_and_confusion(
        gt_json, pred_json, score_threshold=score_threshold, unknown_label="unknown"
    )
    print(f"False Acceptance Rate (FAR) at threshold {score_threshold}: {far_value:.4f}")
    plot_confusion_matrix(far_value,y_true, y_pred, unknown_label="unknown")
