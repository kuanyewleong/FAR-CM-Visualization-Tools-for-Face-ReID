import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict


# ---------- Bounding box IoU ----------
def iou(boxA, boxB):
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])
    return interArea / float(boxAArea + boxBArea - interArea)

# ---------- Load multiple JSON files with reindexing ----------
def load_multiple_json_files(pred_files, gt_files):
    pred_data_all = []
    gt_data = defaultdict(list)
    frame_offset = 0

    # Load GT files with frame reindexing
    for file in gt_files:
        try:
            gt_raw = json.load(file)
            max_frame_id = 0
            for face in gt_raw:
                for f in face["faces"]:
                    local_frame_id = f["frame_id"]
                    new_frame_id = local_frame_id + frame_offset
                    name = f["name"].lower()
                    bbox = {
                        "x1": f["bounding_box"]["top_left"]["x"],
                        "y1": f["bounding_box"]["top_left"]["y"],
                        "x2": f["bounding_box"]["bottom_right"]["x"],
                        "y2": f["bounding_box"]["bottom_right"]["y"],
                    }
                    gt_data[new_frame_id].append({"label": name, "bbox": bbox})
                    max_frame_id = max(max_frame_id, local_frame_id)
            frame_offset += max_frame_id + 1
        except Exception as e:
            st.error(f"Error reading GT file {file.name}: {e}")

    # Reset for prediction frame offsetting
    frame_offset = 0
    for file in pred_files:
        try:
            pred_raw = json.load(file)
            for i, frame in enumerate(pred_raw):
                # Extract numeric frame index from filename (e.g., frame0042.png -> 42)
                match = re.search(r'frame(\d+)', frame["image"])
                if match:
                    local_frame_id = int(match.group(1)) - 1  # Convert to 0-indexed
                else:
                    local_frame_id = i  # fallback

                new_frame_id = local_frame_id + frame_offset
                pred_data_all.append({
                    "frame_id": new_frame_id,
                    "faces": frame["faces"]
                })

            max_local = max([int(re.search(r'frame(\d+)', f["image"]).group(1)) - 1
                             for f in pred_raw if re.search(r'frame(\d+)', f["image"])])
            frame_offset += max_local + 1
        except Exception as e:
            st.error(f"Error reading prediction file {file.name}: {e}")

    # Sort by global frame_id
    pred_data_all = sorted(pred_data_all, key=lambda x: x["frame_id"])
    return pred_data_all, gt_data

# ---------- Analyze ----------
def analyze(pred_data, gt_data, threshold):
    counter = Counter()
    confusion = []
    score_groups = {"Correct": [], "Wrong": [], "Unknown": []}

    for frame in pred_data:
        frame_id = frame["frame_id"]
        pred_faces = frame["faces"]
        gt_faces = gt_data[frame_id]
        gt_matched = [False] * len(gt_faces)

        for pred in pred_faces:
            score = pred["score"]
            pred_label = pred["label"].lower()
            pred_bbox = pred["bbox"]

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_faces):
                if gt_matched[idx]:
                    continue
                iou_val = iou(pred_bbox, gt["bbox"])
                if iou_val >= 0.5 and iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = idx

            if score < threshold:
                if best_gt_idx != -1:
                    gt_label = gt_faces[best_gt_idx]["label"]
                    confusion.append((gt_label, "unknown"))
                    score_groups["Unknown"].append(score)
                    gt_matched[best_gt_idx] = True
                else:
                    confusion.append(("none", "unknown"))
                    score_groups["Unknown"].append(score)
                counter["unknown"] += 1
                continue

            if best_gt_idx != -1:
                gt_label = gt_faces[best_gt_idx]["label"]
                gt_matched[best_gt_idx] = True
                if pred_label == gt_label:
                    counter[pred_label] += 1
                    confusion.append((gt_label, pred_label))
                    score_groups["Correct"].append(score)
                else:
                    counter["wrong matches"] += 1
                    confusion.append((gt_label, pred_label))
                    score_groups["Wrong"].append(score)
            else:
                counter[pred_label] += 1
                confusion.append(("none", pred_label))
                score_groups["Unknown"].append(score)

    return counter, confusion, score_groups

def optimal_bins(data):
    n = len(data)
    return min(max(n // 2, 10), 100) if n > 0 else 10

# FAR calculation
def compute_far(confusion_matrix: pd.DataFrame, unknown_label: str = "unknown") -> float:
    """
    Compute FAR = (All false accepts excluding 'unknown') / (All correct predictions)

    - False accepts: off-diagonal elements, excluding those involving 'unknown' in either axis
    - Correct predictions: diagonal elements (label == label)

    Returns 0.0 if correct predictions == 0
    """
    false_accepts = 0
    correct_predictions = 0

    for gt_label in confusion_matrix.index:
        for pred_label in confusion_matrix.columns:
            val = confusion_matrix.loc[gt_label, pred_label]
            if unknown_label in {gt_label.lower(), pred_label.lower()}:
                continue  # Skip any cell involving 'unknown'
            if gt_label == pred_label:
                correct_predictions += val
            else:
                false_accepts += val

    far = false_accepts / correct_predictions if correct_predictions > 0 else 0.0
    return far


def precompute_for_curves(pred_data, gt_data):
    """
    Build a compact list of (score, correct?, wrong_labeled?, gt_is_none?)
    independent of any threshold.
    """
    scores, correct, wrong_labeled, gt_is_none = [], [], [], []

    for frame in pred_data:
        frame_id = frame["frame_id"]
        pred_faces = frame["faces"]
        gt_faces = gt_data[frame_id]
        gt_matched = [False] * len(gt_faces)

        for pred in pred_faces:
            s = pred["score"]
            pl = pred["label"].lower()
            pb = pred["bbox"]

            # best IoU match
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(gt_faces):
                if gt_matched[idx]:
                    continue
                iv = iou(pb, gt["bbox"])
                if iv >= 0.5 and iv > best_iou:
                    best_iou, best_idx = iv, idx

            if best_idx != -1:
                gt_matched[best_idx] = True
                gl = gt_faces[best_idx]["label"].lower()
                c = (pl == gl)
                wl = (pl != gl)
                gn = False
            else:
                gl = None
                c, wl, gn = False, False, True  # no GT match (row "none" in your CM)

            scores.append(s)
            correct.append(c)
            wrong_labeled.append(wl)
            gt_is_none.append(gn)

    return (np.array(scores, dtype=float),
            np.array(correct, dtype=bool),
            np.array(wrong_labeled, dtype=bool),
            np.array(gt_is_none, dtype=bool))


# Formatting helper
def format_scientific(value):
    formatted = "{:.2e}".format(value)       # e.g. "1.93e-04"
    mantissa, exponent = formatted.split("e")
    exponent = exponent.replace("+0", "+").replace("-0", "-")  # remove leading zeros
    return f"{mantissa}e{int(exponent)}"     # int() removes any extra zero padding


# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("Face Recognition Analysis Tool")

# File upload UI
st.sidebar.header("Upload JSON Files")
pred_files = st.sidebar.file_uploader("Upload Prediction JSON Files", type="json", accept_multiple_files=True)
gt_files = st.sidebar.file_uploader("Upload Ground Truth JSON Files", type="json", accept_multiple_files=True)

threshold = st.slider("Score Threshold", 0.000, 1.000, 0.500, step=0.001, format="%.3f")

if pred_files and gt_files:
    pred_data, gt_data = load_multiple_json_files(pred_files, gt_files)
    label_counts, confusion_pairs, score_groups = analyze(pred_data, gt_data, threshold)

    # ---------- Accuracy & FAR vs Similarity Threshold ----------
    # ---------- (independent of similarity score slider)        
    st.subheader("Accuracy & FAR vs Similarity Threshold")

    scores_np, correct_np, wrong_np, none_np = precompute_for_curves(pred_data, gt_data)

    n_all = scores_np.size
    matched_total = int((1 - none_np).sum())  # number of predictions with a GT match (IoU>=0.5)

    if n_all == 0 or matched_total == 0:
        st.info("Not enough matched predictions to plot Accuracy/FAR vs threshold.")
    else:
        # Sort once by score (desc) and build prefix sums for matched-only metrics
        order = np.argsort(-scores_np)
        s  = scores_np[order]
        c  = correct_np[order]                 # 1 for matched & correct; 0 otherwise
        wl = wrong_np[order]                   # 1 for matched & wrong-label; 0 otherwise

        cum_tp = np.cumsum(c)
        cum_wl = np.cumsum(wl)

        # Threshold grid
        thresholds_grid = np.linspace(0.0, 1.0, 201)
        k = np.searchsorted(-s, -thresholds_grid, side='right')  # how many accepted at each threshold

        TP = np.where(k > 0, cum_tp[k - 1], 0)
        WL = np.where(k > 0, cum_wl[k - 1], 0)

        # --- FAR via compute_far(confusion_matrix) ---
        fars = []
        for tp_i, wl_i in zip(TP, WL):
            # Minimal confusion matrix equivalent to full CM for FAR:
            # diag = tp_i, off-diagonal (excluding 'unknown') = wl_i
            cm = pd.DataFrame([[tp_i, wl_i]], index=['all'], columns=['all', 'other'])
            fars.append(compute_far(cm, unknown_label="unknown"))
        fars = np.array(fars, dtype=float)

        # --- Accuracy (Pred) consistent with your confusion-matrix metric:
        #     correct / total_predicted, where total_predicted counts ONLY matched predictions
        #     (unknowns are included in the denominator as they would be column 'unknown' in the full CM)
        acc_curve = np.where(matched_total > 0, TP / matched_total, 0.0)

        # Plot
        fig_thr, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        # Replace invalid or negative values with exact zeros (FAR line should stay at 0)
        fars_plot = np.nan_to_num(fars, nan=0.0, posinf=0.0, neginf=0.0)

        # Plot FAR and Accuracy
        line_far, = ax2.plot(thresholds_grid, fars_plot, linestyle='--', label="FAR", color='tab:green')
        line_acc, = ax1.plot(thresholds_grid, acc_curve, label="Accuracy")

        # Accuracy value at the intersection of the vertical threshold line and the Accuracy curve ---
        # Find how many predictions are accepted at the current threshold
        idx_thr = np.searchsorted(-s, -threshold, side='right')
        tp_thr = cum_tp[idx_thr - 1] if idx_thr > 0 else 0
        acc_at_thr = (tp_thr / matched_total) if matched_total > 0 else 0.0

        # Mark and annotate the point on the Accuracy curve
        ax1.scatter([threshold], [acc_at_thr], s=36, zorder=20, color='tab:blue', edgecolors='white', linewidths=0.6)
        ax1.annotate(
            f"Acc={acc_at_thr*100:.2f}%",
            xy=(threshold, acc_at_thr),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
            color="tab:blue",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        # Current slider threshold (vertical line)        
        ax1.axvline(threshold, linestyle=':', linewidth=1, color='tab:red')
        x_range = ax1.get_xlim()[1] - ax1.get_xlim()[0]
        offset = -0.005 * x_range  # 0.5% of x-axis range
        ax1.text(
            threshold + offset,  # Dynamic offset based on data range
            ax1.get_ylim()[0] + 0.08 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
            f"thr={threshold:.3f}",
            rotation=90, va='bottom', ha='right', fontsize=8
        )

        ax1.set_xlim(0.0, 1.0)        
        ax1.set_xlabel("Similarity Threshold")
        ax1.set_ylabel("Accuracy (over matched predictions)", color='tab:blue')        
        ax1.set_title("Accuracy & FAR vs Similarity Threshold")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Set x-axis ticks with 0.1 interval
        x_ticks = np.arange(0.0, 1.1, 0.1)
        ax1.set_xticks(x_ticks)

        ax2.set_ylabel("FAR", color='tab:green')
        # Matplotlib can’t apply a log scale if all FAR values are zero
        # Fallback to linear scale if all FAR values are zero
        if np.any(fars_plot > 0):
            ax2.set_yscale('log')
            ax2.set_ylim(bottom=1e-7, top=2e-2)
        else:
            ax2.set_yscale('linear')
            ax2.set_ylim(bottom=0.0, top=1.0)

        far_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        far_labels = ['1e-6', '1e-5', '1e-4', '1e-3', '1e-2']
        ax2.set_yticks(far_ticks)
        ax2.set_yticklabels(far_labels)

        min_log_value = 1e-7
        ax2.text(1.02, min_log_value, '0', transform=ax2.transAxes, 
                ha='left', va='bottom', fontsize=plt.rcParams['ytick.labelsize'])
        ax2.set_ylim(bottom=min_log_value, top=2e-2)

        ax1.set_ylim(bottom=0.0)
        y1_min, y1_max = ax1.get_ylim()

        ax1.spines['bottom'].set_color('gray')
        ax1.spines['bottom'].set_linewidth(0.2)
        ax1.spines['bottom'].set_alpha(0.5)
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

        for line in ax1.get_lines(): line.set_zorder(10)
        for line in ax2.get_lines(): line.set_zorder(10)

        handles = [line_acc, line_far]
        ax1.legend(handles, [h.get_label() for h in handles], loc="best")

        fig_thr.tight_layout()
        st.pyplot(fig_thr)

        st.caption(
            "FAR is computed with accepted (score ≥ threshold) matched predictions. "
            "'unknown' and GT='none' are excluded."
        )


    # Bar chart
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Label Counts")
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, counts)
        ax.set_xlabel("Labels")
        ax.set_ylabel("Count")
        ax.set_title(f"Score Threshold: {threshold:.2f}")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(count), ha='center', va='bottom')
        st.pyplot(fig)

    # Confusion Matrix & FAR
    with col2:
        st.subheader("Confusion Matrix (GT vs Predicted)")

        df = pd.DataFrame(confusion_pairs, columns=["Ground Truth", "Predicted"])
        gt_labels = sorted(set(df["Ground Truth"]) - {"none", "unknown"})
        pred_labels = sorted(set(df["Predicted"]))

        # Move 'unknown' and 'none' to the end for display clarity
        for special in ["unknown", "none"]:
            if special in pred_labels:
                pred_labels.remove(special)
                pred_labels.append(special)

        confusion_matrix = pd.crosstab(
            df["Ground Truth"], df["Predicted"],
            rownames=["GT"], colnames=["Predicted"],
            dropna=False
        ).reindex(index=gt_labels, columns=pred_labels, fill_value=0)

        # Metrics
        total_predicted = confusion_matrix.sum().sum()
        total_gt = confusion_matrix.sum(axis=1).sum()
        correct_predictions = sum(
            confusion_matrix.loc[label, label]
            for label in confusion_matrix.index
            if label in confusion_matrix.columns
        )

        accuracy_pred = correct_predictions / total_predicted if total_predicted > 0 else 0
        accuracy_gt = correct_predictions / total_gt if total_gt > 0 else 0
        far = compute_far(confusion_matrix, unknown_label="unknown")

        # Annotate the matrix
        annotations = confusion_matrix.copy().astype(str)
        row_totals = confusion_matrix.sum(axis=1)
        col_totals = confusion_matrix.sum(axis=0)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                val = confusion_matrix.iloc[i, j]
                gt_label = confusion_matrix.index[i]
                pred_label = confusion_matrix.columns[j]
                row_total = row_totals[gt_label]
                col_total = col_totals[pred_label]
                annotations.iloc[i, j] = f"{val}\n{val}/{row_total} (GT)\n{val}/{col_total} (Pred)"

        # Plot
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        sns.heatmap(confusion_matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax2)
        ax2.set_title(
            f"GT vs Predicted | Acc (Pred): {accuracy_pred:.2%} | Acc (GT): {accuracy_gt:.2%} | FAR: {format_scientific(far)}\n"
            f"Correct: {correct_predictions} / Pred: {total_predicted}, GT: {total_gt}"
        )
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("Ground Truth Label")
        st.pyplot(fig2)


    # Histograms
    st.subheader("Similarity Score Distribution: Raw Count vs Normalized Density")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Raw Count Histogram**")
        fig_raw, ax_raw = plt.subplots(figsize=(8, 4))
        has_data = False
        if score_groups["Correct"]:
            ax_raw.hist(score_groups["Correct"], bins=80, alpha=0.8, color='green', label='Correct Matches')
            has_data = True
        if score_groups["Wrong"]:
            ax_raw.hist(score_groups["Wrong"], bins=50, alpha=0.8, color='red', label='Wrong Matches')
            has_data = True
        if score_groups["Unknown"]:
            ax_raw.hist(score_groups["Unknown"], bins=50, alpha=0.5, color='gray', label='Unknown or Low Similarity')
            has_data = True
        if has_data:
            ax_raw.set_xlim(0.0, 1.0)
            ax_raw.set_xticks([round(x * 0.1, 1) for x in range(11)])
            ax_raw.set_xlabel("Cosine Similarity Score")
            ax_raw.set_ylabel("Count")
            ax_raw.legend()
            st.pyplot(fig_raw)
        else:
            st.info("No score data available for this threshold.")

    bins_correct = optimal_bins(score_groups["Correct"])
    bins_wrong = optimal_bins(score_groups["Wrong"])
    bins_unknown = optimal_bins(score_groups["Unknown"])

    with col4:
        st.markdown("**Normalized Density Histogram**")
        fig_density, ax_density = plt.subplots(figsize=(8, 4))
        has_density_data = False
        if score_groups["Correct"]:
            ax_density.hist(score_groups["Correct"], bins=bins_correct, alpha=0.8, color='green',
                            label='Correct Matches', density=True)
            has_density_data = True
        if score_groups["Wrong"]:
            ax_density.hist(score_groups["Wrong"], bins=bins_wrong, alpha=0.8, color='red',
                            label='Wrong Matches', density=True)
            has_density_data = True
        if score_groups["Unknown"]:
            ax_density.hist(score_groups["Unknown"], bins=bins_unknown, alpha=0.5, color='gray',
                            label='Unknown or Low Similarity', density=True)
            has_density_data = True
        if has_density_data:
            ax_density.set_xlim(0.0, 1.0)
            ax_density.set_xticks([round(x * 0.1, 1) for x in range(11)])
            ax_density.set_xlabel("Cosine Similarity Score")
            ax_density.set_ylabel("Density")
            ax_density.legend()
            st.pyplot(fig_density)
        else:
            st.info("No score data to display in normalized form.")    
else:
    st.info("Please upload **multiple ground truth** and **prediction JSON files** to begin.")
