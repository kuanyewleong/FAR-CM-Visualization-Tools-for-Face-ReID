import json
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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


# Formatting helper
def format_scientific(value):
    formatted = "{:.2e}".format(value)       # e.g. "1.93e-04"
    mantissa, exponent = formatted.split("e")
    exponent = exponent.replace("+0", "+").replace("-0", "-")  # remove leading zeros
    return f"{mantissa}e{int(exponent)}"     # int() removes any extra zero padding


# ---------- Monotone smoothing & inversion helpers for FAR-threshold mapping ----------

def _pava(y, weights=None, increasing=True):
    """
    Simple Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression.
    Returns a vector y_hat of the same length as y.
    If increasing=False, enforces a non-increasing fit.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # For decreasing, flip sign to make it increasing, then flip back
    flip = not increasing
    if flip:
        y = -y

    # Maintain blocks (value, weight, length)
    vals = []
    wts = []
    lens = []
    for i in range(n):
        vals.append(y[i])
        wts.append(weights[i])
        lens.append(1)
        # Merge while previous block > current block (violates monotonicity)
        while len(vals) >= 2 and vals[-2] > vals[-1] + 1e-15:
            new_w = wts[-2] + wts[-1]
            new_v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / new_w
            new_len = lens[-2] + lens[-1]
            vals[-2] = new_v
            wts[-2] = new_w
            lens[-2] = new_len
            vals.pop(); wts.pop(); lens.pop()

    y_hat = np.repeat(vals, lens)
    if flip:
        y_hat = -y_hat
    return y_hat


def fit_far_curve(thresholds, far_values, method="isotonic", eps=1e-12, poly_deg=3, n_points=400):
    """
    Fit a smooth/monotone curve to FAR(threshold).

    - method="isotonic": monotone non-increasing fit in log10(FAR+eps) space (robust default)
    - method="log-linear": piecewise-linear interpolation in log10(FAR+eps), then enforce non-increasing
    - method="poly3": degree-3 polynomial fit in log10(FAR+eps), then enforce non-increasing

    Returns (x_fit, y_fit) with y_fit > 0.
    """
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(far_values, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, eps)  # avoid zeros on log scale

    if method == "isotonic":
        logy = np.log10(y)
        logy_fit = _pava(logy, increasing=False)  # enforce decreasing
        x_fit = x
        y_fit = 10.0 ** logy_fit

    elif method == "log-linear":
        x_fit = np.linspace(x.min(), x.max(), n_points)
        logy = np.log10(y)
        logy_interp = np.interp(x_fit, x, logy)
        y_fit = 10.0 ** logy_interp
        # enforce non-increasing (from left->right) by cumulative minimum from the right
        y_fit = np.minimum.accumulate(y_fit[::-1])[::-1]

    elif method == "poly3":
        logy = np.log10(y)
        coeffs = np.polyfit(x, logy, poly_deg)  # deg=3 by default
        x_fit = np.linspace(x.min(), x.max(), n_points)
        logy_pred = np.polyval(coeffs, x_fit)
        y_fit = 10.0 ** logy_pred
        y_fit = np.minimum.accumulate(y_fit[::-1])[::-1]  # enforce non-increasing

    else:
        x_fit, y_fit = x, y  # fallback (no fitting)

    return x_fit, y_fit


def threshold_for_target_far(
    x_fit, y_fit, target_far, eps=1e-12, extend="clamp"
):
    """
    Invert monotone-decreasing FAR fit y_fit(x_fit) to get threshold for a target FAR.
    extend="clamp"  -> clamp to [x.min, x.max] when target FAR is outside fitted range.
    extend="extrapolate" -> linear extrapolation in log10(FAR) space (then clipped to [x.min, x.max]).
    Returns (thr, status) where status ∈ {"ok","clamped_low","clamped_high","extrap_low","extrap_high"}.
    """
    if target_far is None or not np.isfinite(target_far) or target_far <= 0:
        return None, "invalid"

    y = np.maximum(np.asarray(y_fit, dtype=float), eps)
    x = np.asarray(x_fit, dtype=float)

    logy = np.log10(y)
    logt = np.log10(max(target_far, eps))

    # Reverse so logy_rev is increasing for interpolation
    x_rev = x[::-1]
    logy_rev = logy[::-1]

    # In-range: direct interpolation
    if logy.min() <= logt <= logy.max():
        thr = float(np.interp(logt, logy_rev, x_rev))
        return thr, "ok"

    # Outside range -> clamp or extrapolate
    # Target FAR larger than fitted max FAR -> need smaller threshold (left edge)
    if logt > logy.max():
        if extend == "clamp":
            return float(x.min()), "clamped_low"
        # extrapolate using first two points in (logy_rev, x_rev)
        if len(x_rev) >= 2 and logy_rev[1] != logy_rev[0]:
            slope = (x_rev[1] - x_rev[0]) / (logy_rev[1] - logy_rev[0])
            thr = x_rev[0] + slope * (logt - logy_rev[0])
        else:
            thr = x.min()
        return float(np.clip(thr, x.min(), x.max())), "extrap_low"

    # Target FAR smaller than fitted min FAR -> need larger threshold (right edge)
    if logt < logy.min():
        if extend == "clamp":
            return float(x.max()), "clamped_high"
        # extrapolate using last two points in (logy_rev, x_rev)
        if len(x_rev) >= 2 and logy_rev[-1] != logy_rev[-2]:
            slope = (x_rev[-1] - x_rev[-2]) / (logy_rev[-1] - logy_rev[-2])
            thr = x_rev[-1] + slope * (logt - logy_rev[-1])
        else:
            thr = x.max()
        return float(np.clip(thr, x.min(), x.max())), "extrap_high"


# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("Face Recognition Analysis Tool")

# File upload UI
st.sidebar.header("Upload JSON Files")
pred_files = st.sidebar.file_uploader("Upload Prediction JSON Files", type="json", accept_multiple_files=True)
gt_files = st.sidebar.file_uploader("Upload Ground Truth JSON Files", type="json", accept_multiple_files=True)

# threshold = st.slider("Score Threshold", 0.0, 1.0, 0.5, 0.01)
threshold = 0.5

if pred_files and gt_files:
    pred_data, gt_data = load_multiple_json_files(pred_files, gt_files)
    label_counts, confusion_pairs, score_groups = analyze(pred_data, gt_data, threshold)

    @st.cache_data(show_spinner=False)
    def _precompute_cached_for_thresholds(pred_blob, gt_blob):
        return precompute_for_curves(pred_blob, gt_blob)

    scores_np, correct_np, wrong_np, none_np = _precompute_cached_for_thresholds(pred_data, gt_data)

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


    # ---------- NEW: FAR–Similarity Mapping with Interpolation / Curve Fit ----------
    st.subheader("FAR–Similarity Mapping (Interpolated / Curve Fit)")

    # Fitting options
    fit_col_left, fit_col_right = st.columns([3, 2])
    with fit_col_right:
        method_label = st.selectbox(
            "Fit method",
            ("Isotonic (monotone, default)", "Log-linear (interp)", "Polynomial (deg 3)"),
            help=(
                "• Isotonic: robust monotone fit in log10(FAR) space.\n"
                "• Log-linear: piecewise linear in log10(FAR); then monotone-adjusted.\n"
                "• Polynomial: degree-3 in log10(FAR); then monotone-adjusted."
            )
        )
        method_map = {
            "Isotonic (monotone, default)": "isotonic",
            "Log-linear (interp)": "log-linear",
            "Polynomial (deg 3)": "poly3",
        }
        method = method_map[method_label]

        eps = st.number_input(
            "Numerical epsilon (for log scale)",
            min_value=1e-15, max_value=1e-2, value=1e-12, step=1e-12, format="%.0e",
            help="Used to avoid log(0). Rarely needs changing."
        )
        target_far_str = st.text_input(
            "Conditioned FAR (e.g., 1e-4)", value="1e-4",
            help="Enter in scientific or decimal form, e.g., 0.0001 or 1e-4."
        )
        extend_option = st.selectbox(
            "Extrapolation method",
            ("clamp", "extrapolate"),
            help=(
                "• Clamp: restricts the threshold to the fitted range.\n"
                "• Extrapolate: extends the fitted curve beyond the range."
            )
        )
        extend_map = {
            "clamp": "clamp",
            "extrapolate": "extrapolate",
        }
        extend = extend_map[extend_option]

        try:
            target_far = float(target_far_str)
            if target_far <= 0:
                raise ValueError
        except Exception:
            target_far = None
            st.warning("Please enter a valid positive FAR value (e.g., 1e-4).", icon="⚠️")

    # Fit curve to existing FAR-vs-threshold samples
    x_fit, y_fit = fit_far_curve(thresholds_grid, fars, method=method, eps=eps, poly_deg=3, n_points=600)
    thr_at_target, thr_status = threshold_for_target_far(
        x_fit, y_fit, target_far, eps=eps, extend=extend)

    with fit_col_left:
        fig_fit, ax_fit = plt.subplots(figsize=(9, 5))

        # Plot raw FAR samples and fitted curve
        raw_far = np.maximum(np.nan_to_num(fars, nan=0.0, posinf=0.0, neginf=0.0), eps)
        ax_fit.plot(thresholds_grid, raw_far, linestyle=":", linewidth=1.0, label="FAR (raw samples)")
        ax_fit.plot(x_fit, y_fit, linewidth=2.0, label=f"FAR fit: {method_label}")

        # Current slider threshold annotation
        # ax_fit.axvline(threshold, linestyle=":", linewidth=1, color="tab:red", label=f"Current thr={threshold:.2f}")

        # Target FAR and corresponding threshold (if solvable)
        if target_far is not None:
            ax_fit.axhline(target_far, linestyle="--", linewidth=1, label=f"Conditioned FAR={format_scientific(target_far)}")
        if thr_at_target is not None:
            ax_fit.axvline(thr_at_target, linestyle="--", linewidth=1, label=f"thr @ conditioned FAR ≈ {thr_at_target:.3f}")

        ax_fit.set_xlim(0.0, 1.0)
        ax_fit.set_yscale("log")
        ax_fit.set_xlabel("Similarity Threshold (thr)")
        ax_fit.set_ylabel("FAR (log scale)")
        ax_fit.set_title("Interpolated Mapping: Similarity Threshold → FAR")
        ax_fit.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_fit.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax_fit.legend(loc="best")

        st.pyplot(fig_fit)

    # Compute accuracy_pred at the target threshold: correct_predictions / total_predicted
    # TP at a threshold is given by cum_tp[k-1], and total_predicted is simply n_all    
    acc_pred_at_target = None
    if thr_at_target is not None:
        k_target = np.searchsorted(-s, -thr_at_target, side='right')
        tp_target = int(cum_tp[k_target - 1]) if k_target > 0 else 0
        total_predicted_all = n_all
        acc_pred_at_target = (tp_target / total_predicted_all) if total_predicted_all > 0 else 0.0

    if thr_at_target is not None:
        st.success(
            f"Estimated similarity threshold for FAR **{format_scientific(target_far)}** = **{thr_at_target:.3f}**.\n\n"
            f"At this threshold: **Accuracy (Pred)** ≈ **{acc_pred_at_target:.2%}**.",
            icon="✅"
        )
        if thr_status.startswith("clamped"):
            which = "lowest" if "low" in thr_status else "highest"
            st.caption(f"_Note: Target FAR was outside the fitted range; the threshold was clamped to the **{which}** similarity in the domain._")


    # Offer download of the fitted mapping
    map_df = pd.DataFrame({"threshold": x_fit, "far_fit": y_fit})
    st.download_button(
        label="Download FAR–threshold mapping (CSV)",
        data=map_df.to_csv(index=False).encode("utf-8"),
        file_name="far_threshold_mapping.csv",
        mime="text/csv",
    )

    # --- Save "FAR–Similarity Mapping" plot as PNG ---
    png_cols = st.columns([2, 1, 1, 1])
    with png_cols[1]:
        png_name = st.text_input(
            "PNG filename",
            value="far_threshold_mapping.png",
            help="End with .png"
        )
    with png_cols[2]:
        png_dpi = st.number_input("DPI", min_value=72, max_value=600, value=200, step=1)
    with png_cols[3]:
        png_transparent = st.checkbox("Transparent BG", value=False)

    buf = io.BytesIO()
    fig_fit.savefig(buf, format="png", dpi=png_dpi, bbox_inches="tight", transparent=png_transparent)
    buf.seek(0)

    st.download_button(
        label="Download this plot as PNG",
        data=buf,
        file_name=png_name if png_name.strip().lower().endswith(".png") else f"{png_name.strip()}.png",
        mime="image/png",
    )

