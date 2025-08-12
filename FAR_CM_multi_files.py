import json
import os
from glob import glob

def load_and_reindex_ground_truth(gt_paths):
    all_gt = []
    global_frame = 0

    for path in gt_paths:
        with open(path, "r") as f:
            data = json.load(f)

        for entry in data:
            reindexed_faces = []
            for face in entry["faces"]:
                face["frame_id"] = global_frame
                reindexed_faces.append(face)

            all_gt.append({
                "camera_name": entry.get("camera_name", ""),
                "video_path": entry.get("video_path", ""),
                "faces": reindexed_faces
            })
            global_frame += 1

    return all_gt

def load_and_reindex_predictions(pred_paths):
    all_pred = []
    global_frame = 0

    for path in pred_paths:
        with open(path, "r") as f:
            data = json.load(f)

        for entry in data:
            new_entry = {
                "image": f"frame{global_frame + 1:04d}.png",
                "faces": entry["faces"]
            }
            all_pred.append(new_entry)
            global_frame += 1

    return all_pred

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

def match_predictions_with_gt(gt_faces, pred_faces, iou_threshold=0.3, score_threshold=0.0, unknown_label="unknown"):
    matches = []

    for pred in pred_faces:
        # Apply score thresholding: treat as 'unknown' if below threshold
        pred_label = pred["label"]
        if pred.get("score", 1.0) < score_threshold:
            pred_label = unknown_label

        pred_bbox = {
            "x1": pred["bbox"]["x1"],
            "y1": pred["bbox"]["y1"],
            "x2": pred["bbox"]["x2"],
            "y2": pred["bbox"]["y2"]
        }

        best_iou = 0
        best_gt = None
        for gt in gt_faces:
            gt_bbox = {
                "x1": gt["bounding_box"]["top_left"]["x"],
                "y1": gt["bounding_box"]["top_left"]["y"],
                "x2": gt["bounding_box"]["bottom_right"]["x"],
                "y2": gt["bounding_box"]["bottom_right"]["y"]
            }

            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt = gt

        if best_gt:
            matches.append((pred_label, best_gt["name"]))

    return matches

def calculate_far_from_json(gt_json, pred_json, score_threshold=0.0, unknown_label="unknown"):
    false_accepts = 0
    valid_attempts = 0

    # Build GT dictionary by frame_id
    gt_by_frame = {}
    for entry in gt_json:
        for face in entry["faces"]:
            frame_id = face["frame_id"]
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append(face)

    for entry in pred_json:
        frame_str = entry["image"]  # e.g., "frame0001.png"
        frame_id = int(os.path.splitext(frame_str)[0].replace("frame", "")) - 1

        if frame_id not in gt_by_frame:
            continue

        pred_faces = entry["faces"]
        gt_faces = gt_by_frame[frame_id]

        matches = match_predictions_with_gt(
            gt_faces, pred_faces,
            iou_threshold=0.3,
            score_threshold=score_threshold,
            unknown_label=unknown_label
        )

        for pred_label, gt_label in matches:
            pred_label_lower = pred_label.lower()
            gt_label_lower = gt_label.lower()
            if pred_label_lower == unknown_label.lower():
                continue
            valid_attempts += 1
            if pred_label_lower != gt_label_lower:
                false_accepts += 1

    far = (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0
    return far


if __name__ == "__main__":
    ground_truth_files = sorted(glob("ground_truth/5pc/right/*.json"))  # Replace with actual paths
    prediction_files = sorted(glob("analysis_results/ir101_webface12m/5pc_right/*.json"))  # Replace with actual paths

    gt_json = load_and_reindex_ground_truth(ground_truth_files)
    pred_json = load_and_reindex_predictions(prediction_files)

    # User can set the score threshold here
    score_threshold = 0.9

    far_value = calculate_far_from_json(
        gt_json, pred_json,
        score_threshold=score_threshold,
        unknown_label="unknown"
    )
    print(f"False Acceptance Rate (FAR) at threshold {score_threshold}: {far_value:.4f}")


