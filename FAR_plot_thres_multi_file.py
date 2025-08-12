import json
import os
import matplotlib.pyplot as plt
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

def match_predictions_with_gt(gt_faces, pred_faces, iou_threshold=0.3):
    matches = []
    for pred in pred_faces:
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
            matches.append((pred["label"], best_gt["name"], pred.get("score", 1.0)))
    
    return matches

def calculate_far(gt_json, pred_json, unknown_label="unknown", score_threshold=0.0):
    false_accepts = 0
    valid_attempts = 0

    gt_by_frame = {}
    for entry in gt_json:
        for face in entry["faces"]:
            frame_id = face["frame_id"]
            gt_by_frame.setdefault(frame_id, []).append(face)

    for entry in pred_json:
        frame_str = entry["image"]
        frame_id = int(os.path.splitext(frame_str)[0].replace("frame", "")) - 1

        if frame_id not in gt_by_frame:
            continue

        pred_faces = entry["faces"]
        gt_faces = gt_by_frame[frame_id]
        matches = match_predictions_with_gt(gt_faces, pred_faces)

        for pred_label, gt_label, score in matches:
            pred_label_lower = pred_label.lower()
            gt_label_lower = gt_label.lower()

            if pred_label_lower == unknown_label.lower():
                continue
            if score < score_threshold:
                continue

            valid_attempts += 1
            if pred_label_lower != gt_label_lower:
                false_accepts += 1

    return (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0

# ------------------ Load data ------------------

# Load JSON files
ground_truth_files = sorted(glob("ground_truth/3pc/right/*.json"))  # Replace with actual paths
prediction_files = sorted(glob("analysis_results/ir18_casia/3pc_right/*.json"))  # Replace with actual paths

gt_json = load_and_reindex_ground_truth(ground_truth_files)
pred_json = load_and_reindex_predictions(prediction_files)

# ------------------ Compute and Plot FAR Curve ------------------
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

thresholds = [round(x, 2) for x in list(frange(0.0, 1.01, 0.05))]
far_values = []

for t in thresholds:
    far = calculate_far(gt_json, pred_json, unknown_label="unknown", score_threshold=t)
    far_values.append(far)
    print(f"Threshold {t:.2f} => FAR: {far:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, far_values, marker='o')
plt.title("FAR vs. Similarity Score Threshold")
plt.xlabel("Similarity Score Threshold")
plt.ylabel("False Acceptance Rate (FAR)")
plt.grid(True)
plt.xticks(thresholds)

y_max = max(far_values)
if y_max == 0:
    plt.ylim(0, 1)
else:
    plt.ylim(0, y_max * 1.1)

plt.tight_layout()
plt.show()
