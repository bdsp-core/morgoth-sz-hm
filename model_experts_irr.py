import os, glob
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import itertools
import statsmodels.stats.inter_rater as irr
from tqdm import tqdm
import  math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re, ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import shutil

import re, ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc

from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists"""
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)



def combine_experts_model_results_pd(pd_train_test_list_path, expert_results_path,expert_cols, model_results_dir,out_path):
    def majority_or_none(row, col):
        """Return unique mode; return None on tie"""
        vals = [row[i] for i in col]
        cnt = Counter(vals)
        top = cnt.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return top[0][0]

    def choose_prob(row):
        if row['majority'] == 'left':
            return row['l_prob']
        elif row['majority'] == 'right':
            return row['r_prob']
        elif row['majority'] == 'both':
            return row['b_prob']
        elif row['majority'] == 'other':
            return max(row['r_prob'], row['l_prob'], row['b_prob'])
        else:
            return None  # Handle tie/null case

    def assign_true(val):
        if val is None:
            return None
        elif val == "other":
            return 0
        else:
            return 1

    ensure_dir(out_path)

    if expert_results_path.endswith('.csv'):
        results_df=pd.read_csv(expert_results_path)
    elif expert_results_path.endswith('.xlsx'):
        results_df=pd.read_excel(expert_results_path)
    else:
        return

    results_df = results_df.replace("generalized", "both")
    results_df = results_df.replace("bi-lateral", "both")
    results_df['morgoth_pred'] = ''  # Initialize
    results_df['r_prob']=''
    results_df['l_prob'] = ''
    results_df['b_prob'] = ''

    for idx, row in tqdm(results_df.iterrows(), total=results_df.shape[0]):
        file_name = row['File']
        file_path = os.path.join(model_results_dir, f'{file_name}.csv')

        if os.path.exists(file_path):
            model_results_df = pd.read_csv(file_path)

            # Take the middle 5 rows
            n = len(model_results_df)
            if n == 0:
                results_df.at[idx, 'morgoth_pred'] = None
                continue

            mid_start = max(0, n // 2 - 2)
            mid_end = min(n, mid_start + 5)
            mid_df = model_results_df.iloc[mid_start:mid_end]

            # Evaluate conditions
            has_r = (mid_df['r_pred'] >= 0.5).any()
            has_l = (mid_df['l_pred'] >= 0.5).any()

            if has_r and not has_l:
                label = "right"
            elif has_l and not has_r:
                label = "left"
            elif not has_r and not has_l:
                label = "other"
            else:  # has_r and has_l
                label = "both"

            results_df.at[idx, 'morgoth_pred'] = label

            r_mean = mid_df['r_pred'].mean()
            l_mean = mid_df['l_pred'].mean()

            results_df.at[idx, 'r_prob'] = r_mean
            results_df.at[idx, 'l_prob'] = l_mean
            results_df.at[idx, 'b_prob'] = min(r_mean, l_mean)

        else:
            results_df.at[idx, 'morgoth_pred'] = None
            print(f'no results for {file_name}')

    results_df['majority'] = results_df.apply(lambda row: majority_or_none(row, expert_cols), axis=1)

    for i in expert_cols:
        other_cols = [c for c in expert_cols if c != i]
        col_name = f"majority_without_{i}"
        results_df[col_name] = results_df.apply(
            lambda row: majority_or_none(row, other_cols),
            axis=1
        )

    results_df['true'] = results_df['majority'].apply(assign_true)

    results_df['prob'] = results_df.apply(choose_prob, axis=1)

    results_df['BDSPPatientID'] = (
        results_df['File']
        .str.split('_').str[0]  # Split and take the first segment
        .str[9:]  # Slice from the 9th character
    )

    pd_train_test_list_df= pd.read_csv(pd_train_test_list_path)
    pd_train_test_list_df.drop(columns=['Feature','Folder'], inplace=True)
    # Sort by priority: put train first
    pd_train_test_list_df = (
        pd_train_test_list_df
        .sort_values(by="PD", key=lambda col: col.eq("train"), ascending=False)
        .drop_duplicates(subset="BDSPPatientID", keep="first")
    )

    # Unify primary key as string
    results_df['BDSPPatientID'] = results_df['BDSPPatientID'].astype(str).str.strip()
    pd_train_test_list_df['BDSPPatientID'] = pd_train_test_list_df['BDSPPatientID'].astype(str).str.strip()

    # (Optional) Further cleanup of .0 suffixes, whitespace, case differences, etc.
    for df_ in (results_df, pd_train_test_list_df):
        df_['BDSPPatientID'] = (
            df_['BDSPPatientID']
            .str.replace(r'\.0$', '', regex=True)  # Remove trailing .0
            .str.replace(r'\s+', '', regex=True)  # Remove internal whitespace
        )

    # Then merge
    results_df = results_df.merge(
        pd_train_test_list_df, how='left', on='BDSPPatientID'
    )


    results_df.drop(columns=['file'], inplace=True)

    results_df=results_df[results_df['PD']!='train']

    print(len(results_df))

    results_df.to_csv(out_path, index=False)


def find_windows(event_point, win, stride, n):
    """
    Find all window indices (0-based) containing event_point
    :param event_point: event point (e.g. 60525)
    :param win: window size (e.g. 200)
    :param stride: stride (e.g. 8)
    :param n: total result length
    :return: list of indices
    """
    k_min = math.ceil((event_point - win) / stride)
    k_max = math.floor(event_point / stride)
    # Clamp to result range
    k_min = max(k_min, 0)
    k_max = min(k_max, n - 1)

    return list(range(k_min, k_max + 1))


def combine_experts_model_results_vw(expert_results_path,model_results_dir,result_stride,vw_label_path,out_path):
    ensure_dir(out_path)

    if expert_results_path.endswith('.csv'):
        results_df=pd.read_csv(expert_results_path)
    elif expert_results_path.endswith('.xlsx'):
        results_df=pd.read_excel(expert_results_path)
    else:
        return

    if vw_label_path.endswith('.csv'):
        vw_label_df = pd.read_csv(vw_label_path)
    elif vw_label_path.endswith('.xlsx'):
        vw_label_df = pd.read_excel(vw_label_path)
    else:
        return


    results_df = results_df.replace("ied", "other")
    results_df['morgoth_pred'] = ''  # Initialize

    for idx, row in tqdm(results_df.iterrows(), total=results_df.shape[0]):
        file_name = row['File']
        type=row['Feature']
        file_path = os.path.join(model_results_dir, f'{file_name}.csv')

        if os.path.exists(file_path):
            model_results_df = pd.read_csv(file_path)

            n = len(model_results_df)
            if n == 0:
                results_df.at[idx, 'morgoth_pred'] = None
                continue

            if type!='VW':
                # Take the middle 5 rows
                mid_start = max(0, n // 2 - 2)
                mid_end = min(n, mid_start + 5)
                mid_df = model_results_df.iloc[mid_start:mid_end]
                prob = model_results_df.at[n // 2, 'pred']

            else:
                center=int(vw_label_df[vw_label_df['file']==file_name]['event_time_index'])
                stride = result_stride
                win = 200
                result_indx=find_windows(center, win, stride, n)
                mid_df = model_results_df.iloc[result_indx]
                prob = mid_df['pred'].max()

            # Evaluate conditions
            has_vw = (mid_df['pred'] >= 0.5).any()

            if has_vw:
                label = "vw"
            else:
                label = "other"

            results_df.at[idx, 'morgoth_pred'] = label
            results_df.at[idx, 'morgoth_prob'] = prob

        else:
            results_df.at[idx, 'morgoth_pred'] = ""
            print(f'no results for {file_name}')

    results_df['morgoth_pred'] = results_df['morgoth_pred'].where(
        results_df['morgoth_pred'] != "",  # Keep original value if not empty string
        results_df['tyz_label']  # Fill with tyz_label if empty string
    )

    results_df.to_csv(out_path, index=False)



def compute_irr(
    file_path,
    cols,
    new_label_cols_name=None,
    *,
    thresholds=None,   # dict: {'colA':0.5, ...} or float/int single threshold; used to binarize probabilities
    valid_labels=None  # optional: explicitly restrict the allowed label set (e.g. {0,1} or {'L','R','G','N'})
):
    """
    Read file_path and compute across cols:
      1) Pairwise Cohen's kappa
      2) Each column vs majority of others (skip sample on tie)
      3) Fleiss' kappa (overall agreement)
    Key handling:
      - Automatically drop NaN (pairwise/rowwise)
      - If a column contains continuous values, thresholds must be provided for binarization, otherwise raises error
      - Unify string/numeric values as discrete categories
    """
    df = pd.read_csv(file_path)

    if new_label_cols_name is None or len(new_label_cols_name) == 0:
        new_label_cols_name = cols
    if len(new_label_cols_name) != len(cols):
        raise ValueError("Length of new_label_cols_name must match cols.")

    # --------- Utility: check if a column is "continuous" and discretize ----------
    def _is_continuous_series(s: pd.Series) -> bool:
        # Rough check for continuous: many unique values or non-{0.0,1.0} decimals
        uniq = pd.unique(s.dropna())
        if len(uniq) > 20:
            return True
        # If numeric and contains non-integer decimals/probabilities
        nums = pd.to_numeric(pd.Series(uniq), errors='coerce')
        if nums.notna().sum() == len(uniq):
            # Allow integer classes {0,1,...,K}; otherwise treat as continuous
            if np.all(np.mod(nums, 1) == 0):
                return False
            # {0.0,1.0} only is also considered discrete
            if set(np.round(nums, 6)) <= {0.0, 1.0}:
                return False
            return True
        return False  # String category

    def _binarize_by_threshold(s: pd.Series, thr: float):
        return (pd.to_numeric(s, errors='coerce') >= thr).astype(int)

    def _coerce_discrete(s: pd.Series, colname: str) -> pd.Series:
        # Already discrete (few unique values/integers/0-1) -> keep; continuous -> needs threshold
        if _is_continuous_series(s):
            # Threshold must be provided
            col_thr = None
            if isinstance(thresholds, dict):
                col_thr = thresholds.get(colname, None)
            elif thresholds is not None:
                col_thr = float(thresholds)
            if col_thr is None:
                raise ValueError(
                    f"Column {colname!r} appears to be continuous (e.g. probabilities). Please provide thresholds (scalar or dict) for binarization."
                )
            return _binarize_by_threshold(s, col_thr)
        else:
            # Convert to string or integer category; keep as-is but ensure non-NaN
            return s

    # Discretize all columns first (without modifying df, generate a copy)
    disc = {}
    for c in cols:
        disc[c] = _coerce_discrete(df[c], c)

    # Optional: restrict to valid label set
    if valid_labels is not None:
        valid_labels = set(valid_labels)
        for c in cols:
            mask_valid = disc[c].isin(valid_labels)
            disc[c] = disc[c].where(mask_valid, np.nan)

    # =============== 1) pairwise kappa ===============
    pairwise = {}
    for (i, col1), (j, col2) in itertools.combinations(enumerate(cols), 2):
        s1 = pd.Series(disc[col1])
        s2 = pd.Series(disc[col2])
        m = s1.notna() & s2.notna()
        y1 = s1[m]
        y2 = s2[m]
        if len(y1) == 0:
            kappa = np.nan
        else:
            kappa = cohen_kappa_score(y1, y2)
        pairwise[f"{new_label_cols_name[i]}_vs_{new_label_cols_name[j]}"] = kappa

    # =============== 2) vs majority ==================
    majority = {}
    for idx, col in enumerate(cols):
        other_cols = [c for c in cols if c != col]

        y_true, y_pred = [], []
        # When voting for a row from "others", skip NaN
        others_df = pd.DataFrame({c: disc[c] for c in other_cols})
        self_s = pd.Series(disc[col])

        for ridx, row in others_df.iterrows():
            row_vals = row.dropna()
            if row_vals.empty:
                continue
            counter = Counter(row_vals)
            most_common = counter.most_common()
            # Skip on tie
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                continue
            majority_label = most_common[0][0]
            # This column itself must also be non-NaN
            if pd.isna(self_s.iloc[ridx]):
                continue
            y_true.append(majority_label)
            y_pred.append(self_s.iloc[ridx])

        if len(y_true) == 0:
            majority[new_label_cols_name[idx]] = {"kappa": None, "n": 0}
        else:
            kappa = cohen_kappa_score(y_pred, y_true)
            majority[new_label_cols_name[idx]] = {"kappa": kappa, "n": len(y_true)}

    # =============== 3) Fleiss' kappa ================
    # Use only rows where all values are non-NaN
    disc_df = pd.DataFrame({c: disc[c] for c in cols})
    disc_df_drop = disc_df.dropna(how='any')
    if disc_df_drop.empty:
        fleiss = np.nan
    else:
        ratings = disc_df_drop.to_numpy()
        # Compute the full set of categories
        cats = sorted(pd.unique(ratings.ravel()))
        # Generate per-row category counts
        m_counts = []
        for row in ratings:
            m_counts.append([np.sum(row == c) for c in cats])
        fleiss = irr.fleiss_kappa(m_counts)

    return {
        "pairwise": pairwise,
        "vs_majority": majority,
        "fleiss": fleiss
    }


def plot_pairwise_heatmap(results, cols, new_label_cols_name,out_path="irr_pairwise_heatmap.png"):
    ensure_dir(out_path)

    """Plot pairwise Cohen's kappa heatmap based on results['pairwise']"""
    K = len(cols)
    mat = np.full((K, K), np.nan, dtype=float)

    # Reconstruct symmetric matrix from pairwise dict
    pairwise = results["pairwise"]
    for i in range(K):
        for j in range(K):
            if i == j:
                mat[i, j] = 1.0  # Self-comparison set to 1 (can change to np.nan)
            elif i < j:
                key = f"{cols[i]}_vs_{cols[j]}"
                if key in pairwise:
                    mat[i, j] = pairwise[key]
                    mat[j, i] = pairwise[key]
                else:
                    # Also try reversed key name
                    rkey = f"{cols[j]}_vs_{cols[i]}"
                    if rkey in pairwise:
                        mat[i, j] = pairwise[rkey]
                        mat[j, i] = pairwise[rkey]

    fig, ax = plt.subplots(figsize=(1.2*len(cols), 1.0*len(cols)))
    im = ax.imshow(mat, vmin=0, vmax=1)  # kappa commonly in [0,1]; change vmin=-1 if negative

    # Axes and ticks
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(new_label_cols_name, rotation=45, ha="right")
    ax.set_yticklabels(new_label_cols_name)

    # Value annotations
    for i in range(K):
        for j in range(K):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's kappa")
    ax.set_title("Pairwise IRR (Cohen's kappa)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")




def plot_vs_majority_squares(
    results,
    expert_cols,
    new_label_cols_name,
    out_path="figures/irr_vs_majority_squares.png"
):
    """
    Plot each rater vs majority of others in a single column:
    - x-axis is fixed at 'rater'
    - All raters' results are stacked with different colored squares
    - Each point is annotated with new label name + kappa value
    - expert_cols: list of original column names
    - new_label_cols_name: corresponding new name list (must match length of expert_cols)
    """
    ensure_dir(out_path)

    vs_major = results["vs_majority"]
    raters = list(vs_major.keys())
    kappas = [vs_major[r]["kappa"] for r in raters]

    # Build mapping from original name to new name
    name_map = dict(zip(expert_cols, new_label_cols_name))

    fig, ax = plt.subplots(figsize=(5, 10))
    x_pos = 0  # All points plotted in the same column
    colors = plt.cm.tab10.colors  # Color palette

    for i, (r, k) in enumerate(zip(raters, kappas)):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            continue
        # Use the mapped new name
        display_name = name_map.get(r, r)
        ax.scatter([x_pos], [k], marker='s', s=160,
                   color=colors[i % len(colors)], label=display_name)

        # Bold if display_name is not in new name list
        if display_name not in new_label_cols_name:
            fontweight = "bold"
        else:
            fontweight = "normal"

        ax.text(
            x_pos + 0.005, k - 0.005,
            f"{display_name}: {k:.2f}",
            ha="left", va="bottom",
            fontweight=fontweight
        )

    # Set axes
    ax.set_xticks([x_pos])
    ax.set_xticklabels(["rater"])
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0, 1)  # Change to (-1, 1) if negative values are possible
    ax.set_title("Each rater vs majority of others (ties skipped)")
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")




def majority_or_none(vals):
    """Unique mode; return None on tie"""
    c = Counter(vals)
    if not c:
        return None
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]

def _percentile_band(curves, grid, lo=2.5, hi=97.5):
    stack = np.vstack(curves)  # (B, len(grid))
    low = np.percentile(stack, lo, axis=0)
    high = np.percentile(stack, hi, axis=0)
    med = np.percentile(stack, 50, axis=0)
    return low, med, high

# --- Bootstrap for a single expert (hard labels) to get (FPR,TPR,Prec,Rec) CIs ---
def _expert_point_and_ci(y_true_bin, y_pred_bin, n_boot=1000, rng=None):
    """
    Input: y_true(0/1) and y_pred(0/1) after this expert vs majority of others
    Output: point estimate + bootstrap 95% CI (2.5%~97.5%) for FPR, TPR, Precision, Recall
    """
    y_true_bin = np.asarray(y_true_bin, int)
    y_pred_bin = np.asarray(y_pred_bin, int)
    n = len(y_true_bin)
    if n == 0:
        return dict(n=0, fpr=np.nan, tpr=np.nan, prec=np.nan, rec=np.nan,
                    fpr_ci=(np.nan, np.nan), tpr_ci=(np.nan, np.nan),
                    prec_ci=(np.nan, np.nan), rec_ci=(np.nan, np.nan))

    # Point estimates
    TP = int(((y_true_bin==1)&(y_pred_bin==1)).sum())
    FP = int(((y_true_bin==0)&(y_pred_bin==1)).sum())
    TN = int(((y_true_bin==0)&(y_pred_bin==0)).sum())
    FN = int(((y_true_bin==1)&(y_pred_bin==0)).sum())
    tpr  = TP / (TP + FN) if (TP+FN)>0 else np.nan
    fpr  = FP / (FP + TN) if (FP+TN)>0 else np.nan
    prec = TP / (TP + FP) if (TP+FP)>0 else np.nan
    rec  = tpr

    # Bootstrap
    if rng is None:
        rng = np.random.default_rng(2025)
    fprs, tprs, precs, recs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_bin[idx]; yp = y_pred_bin[idx]
        TPb = int(((yt==1)&(yp==1)).sum())
        FPb = int(((yt==0)&(yp==1)).sum())
        TNb = int(((yt==0)&(yp==0)).sum())
        FNb = int(((yt==1)&(yp==0)).sum())
        tpr_b  = TPb / (TPb + FNb) if (TPb+FNb)>0 else np.nan
        fpr_b  = FPb / (FPb + TNb) if (FPb+TNb)>0 else np.nan
        prec_b = TPb / (TPb + FPb) if (TPb+FPb)>0 else np.nan
        rec_b  = tpr_b
        tprs.append(tpr_b); fprs.append(fpr_b)
        precs.append(prec_b); recs.append(rec_b)

    def ci(arr):
        arr = np.asarray(arr, float)
        arr = arr[~np.isnan(arr)]
        if len(arr)==0: return (np.nan, np.nan)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return dict(
        n=n, fpr=fpr, tpr=tpr, prec=prec, rec=rec,
        fpr_ci=ci(fprs), tpr_ci=ci(tprs), prec_ci=ci(precs), rec_ci=ci(recs)
    )


# - Generate all expert points (LOO) and their CIs ---
def _expert_points_loo_with_ci(label, df, expert_cols,new_label_cols_name=[], n_boot=1000, rng=None):
    pts = []
    for idx, r in enumerate(expert_cols):
        others = [c for c in expert_cols if c != r]
        y_true, y_pred = [], []
        for _, row in df[expert_cols].iterrows():
            maj = majority_or_none([row[c] for c in others])
            if maj is None:  # Other experts tied -> skip
                continue
            y_true.append(1 if maj == label else 0)
            y_pred.append(1 if row[r] == label else 0)
        stat = _expert_point_and_ci(y_true, y_pred, n_boot=n_boot, rng=rng)
        stat["name"] = new_label_cols_name[idx]
        pts.append(stat)
    return pts

# --- Main function: overlay expert points + cross CI ---
def plot_roc_pr_with_bootstrap(
    label,
    file_path,
    expert_cols,
    prob_col,
    n_boot,
    out_dir,
new_label_cols_name=[]
):
    ensure_dir(out_dir)

    """
    Read CSV, generate y_true (other=0, vw=1) via multi-expert majority vote (skip ties),
    use prob_col as y_score, plot ROC and PR curves,
    and perform bootstrap to generate shaded confidence bands (2.5%-97.5%).
    Also overlay expert points (LOO) with their bootstrap 95% CIs shown as cross markers.
    """
    df = pd.read_csv(file_path)

    # 1) Majority vote (skip ties)
    majority, keep_idx = [], []
    for i, row in df[expert_cols].iterrows():
        maj = majority_or_none(row.tolist())
        if maj is not None:
            majority.append(maj)
            keep_idx.append(i)
    if not keep_idx:
        raise ValueError("No usable samples: all expert votes are tied.")

    df_use = df.loc[keep_idx].copy()
    if prob_col not in df_use.columns:
        raise ValueError(f"Probability column not found: {prob_col}")

    # Convert labels: other -> 0, positive -> 1
    y_true = np.array([1 if x == label else 0 for x in majority], dtype=int)
    y_score = df_use[prob_col].astype(float).to_numpy()

    # Remove NaN (y_true is 0/1 only; this guards against NaN in y_score)
    mask = ~(np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    n_used = len(y_true)
    if n_used == 0:
        raise ValueError("No usable samples after removing NaN.")

    # 2) Full-data curves
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fpr_grid = np.linspace(0, 1, 201)
    rec_grid = np.linspace(0, 1, 201)
    tpr_on_grid = np.interp(fpr_grid, fpr, tpr)
    prec_on_grid = np.interp(rec_grid, rec, prec)

    # 3) Bootstrap (model curves)
    rng = np.random.default_rng(2025)
    tpr_curves, prec_curves = [], []
    aucs, aps = [], []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx];
        ys = y_score[idx]

        # ---- Skip iteration if only one class present ----
        if yt.max() == yt.min():  # all 0 or all 1
            continue

        # ROC
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        aucs.append(auc(fpr_b, tpr_b))
        tpr_curves.append(np.interp(fpr_grid, fpr_b, tpr_b))

        # PR
        prec_b, rec_b, _ = precision_recall_curve(yt, ys)
        # ---- Deduplicate to ensure rec is strictly increasing before interpolation ----
        rec_b, uniq_idx = np.unique(rec_b, return_index=True)
        prec_b = prec_b[uniq_idx]

        prec_curves.append(np.interp(rec_grid, rec_b, prec_b))
        aps.append(average_precision_score(yt, ys))

    # ---- Warn if no valid curves ----
    if len(prec_curves) == 0 or len(tpr_curves) == 0:
        raise ValueError("No valid curves after bootstrap. Possibly extreme class imbalance; increase sample size or reduce n_boot.")

    roc_lo, roc_med, roc_hi = _percentile_band(tpr_curves, fpr_grid)
    pr_lo, pr_med, pr_hi = _percentile_band(prec_curves, rec_grid)

    # 4) Expert points + CI (LOO)
    expert_pts = _expert_points_loo_with_ci(label,df, expert_cols,new_label_cols_name=new_label_cols_name, n_boot=n_boot, rng=rng)

    # 5) Plot: ROC
    plt.figure(figsize=(5.5, 5.0))
    plt.fill_between(fpr_grid, roc_lo, roc_hi, alpha=0.25, label="Model 95% CI (bootstrap)")
    # plt.plot(fpr_grid, roc_med, lw=1.5, label="Model bootstrap median")
    plt.plot(fpr_grid, tpr_on_grid, lw=2, label=f"Morgoth VW (AUC {roc_auc:.3f})")


    colors = plt.cm.tab10.colors
    for i, p in enumerate(expert_pts):
        if np.isnan(p["fpr"]) or np.isnan(p["tpr"]):
            continue

        # Cross CI: horizontal = FPR CI, vertical = TPR CI
        if not np.isnan(p["fpr"]) and not np.isnan(p["tpr"]):
            fpr_lo, fpr_hi = p["fpr_ci"]
            tpr_lo, tpr_hi = p["tpr_ci"]
            # Horizontal line
            if not np.isnan(fpr_lo) and not np.isnan(fpr_hi):
                plt.plot([fpr_lo, fpr_hi], [p["tpr"], p["tpr"]], color="gray", lw=1.2)
            # Vertical line
            if not np.isnan(tpr_lo) and not np.isnan(tpr_hi):
                plt.plot([p["fpr"], p["fpr"]], [tpr_lo, tpr_hi], color="gray", lw=1.2)

        color = colors[i % len(colors)]
        # Point
        plt.scatter(p["fpr"], p["tpr"], s=65, color=color, label=f"{p['name']} (n={p['n']})",marker="^")

    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{label} ROC with expert points")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    roc_path = os.path.join(out_dir,f"{label}_roc.png")
    plt.savefig(roc_path, dpi=300); plt.close()

    # 6) Plot: PR
    plt.figure(figsize=(5.5, 5.0))
    # === 6) Plot: PR, modify fill_between as: ===
    mask = ~(np.isnan(pr_lo) | np.isnan(pr_hi) | np.isnan(rec_grid))
    if mask.any():
        plt.fill_between(rec_grid[mask], pr_lo[mask], pr_hi[mask],
                         alpha=0.25, label="Model 95% CI (bootstrap)")
    else:
        print("[Warn] PR CI is empty after masking NaNs.")

    # plt.plot(rec_grid, pr_med, lw=1.5, label="Model bootstrap median")  # Uncomment to show median line
    plt.plot(rec, prec, lw=2, label=f"Model (AP {ap:.3f})")


    for i, p in enumerate(expert_pts):
        if np.isnan(p["rec"]) or np.isnan(p["prec"]):
            continue
        # Cross CI: horizontal = Recall CI, vertical = Precision CI
        rec_lo, rec_hi = p["rec_ci"]
        prec_lo, prec_hi = p["prec_ci"]
        if not np.isnan(rec_lo) and not np.isnan(rec_hi):
            plt.plot([rec_lo, rec_hi], [p["prec"], p["prec"]], color="gray", lw=1.2)
        if not np.isnan(prec_lo) and not np.isnan(prec_hi):
            plt.plot([p["rec"], p["rec"]], [prec_lo, prec_hi], color="gray", lw=1.2)

        color = colors[i % len(colors)]
        plt.scatter(p["rec"], p["prec"], s=65, color=color, label=f"{p['name']} (n={p['n']})",marker="^")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{label} PR with expert points")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    pr_path = os.path.join(out_dir,f"{label}_pr.png")
    plt.savefig(pr_path, dpi=300); plt.close()

    # Summary
    auc_ci = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
    ap_ci  = (np.percentile(aps, 2.5),  np.percentile(aps, 97.5))

    print(f"[Saved] {roc_path} | AUC={auc(fpr, tpr):.4f} 95%CI[{auc_ci[0]:.4f},{auc_ci[1]:.4f}]")
    print(f"[Saved] {pr_path}  |  AP={ap:.4f}   95%CI[{ap_ci[0]:.4f},{ap_ci[1]:.4f}]")

    return {
        "n_used": int(n_used),
        "auc": float(auc(fpr, tpr)), "auc_ci": (float(auc_ci[0]), float(auc_ci[1])),
        "ap": float(ap), "ap_ci": (float(ap_ci[0]), float(ap_ci[1])),
        "roc_path": roc_path, "pr_path": pr_path,
        "expert_points": expert_pts
    }




def _bin_label_is_pd(x):
    """Non-'other' -> 1, 'other' -> 0; return None for None/NaN"""
    if pd.isna(x):
        return None
    return 0 if str(x) == 'other' else 1

def _expert_point_and_ci_from_cols(df, expert_col, loo_col, n_boot=1000, rng=None):
    """
    Generate a point + bootstrap CI from an expert column and a LOO majority column
      - Positive class: label != 'other'
      - Skip samples where any column is missing or LOO is tied/missing
    Returns: dict(name, n, fpr,tpr,prec,rec, and their 95%CI)
    """
    # Select rows where both columns are non-null
    sub = df[[expert_col, loo_col]].copy()
    sub = sub.dropna(how='any')
    if sub.empty:
        return dict(name=expert_col, n=0, fpr=np.nan, tpr=np.nan, prec=np.nan, rec=np.nan,
                    fpr_ci=(np.nan,np.nan), tpr_ci=(np.nan,np.nan),
                    prec_ci=(np.nan,np.nan), rec_ci=(np.nan,np.nan))

    y_true = sub[loo_col].map(_bin_label_is_pd)
    y_pred = sub[expert_col].map(_bin_label_is_pd)

    # Drop mapping failures (None)
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask].astype(int).to_numpy()
    yp = y_pred[mask].astype(int).to_numpy()
    n = len(yt)
    if n == 0:
        return dict(name=expert_col, n=0, fpr=np.nan, tpr=np.nan, prec=np.nan, rec=np.nan,
                    fpr_ci=(np.nan,np.nan), tpr_ci=(np.nan,np.nan),
                    prec_ci=(np.nan,np.nan), rec_ci=(np.nan,np.nan))

    # Point estimates
    TP = int(((yt==1)&(yp==1)).sum())
    FP = int(((yt==0)&(yp==1)).sum())
    TN = int(((yt==0)&(yp==0)).sum())
    FN = int(((yt==1)&(yp==0)).sum())
    tpr  = TP / (TP + FN) if (TP+FN)>0 else np.nan
    fpr  = FP / (FP + TN) if (FP+TN)>0 else np.nan
    prec = TP / (TP + FP) if (TP+FP)>0 else np.nan
    rec  = tpr

    # bootstrap CI
    if rng is None:
        rng = np.random.default_rng(2025)
    fprs, tprs, precs, recs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]; yp_b = yp[idx]
        TPb = int(((yt_b==1)&(yp_b==1)).sum())
        FPb = int(((yt_b==0)&(yp_b==1)).sum())
        TNb = int(((yt_b==0)&(yp_b==0)).sum())
        FNb = int(((yt_b==1)&(yp_b==0)).sum())
        tpr_b  = TPb / (TPb + FNb) if (TPb+FNb)>0 else np.nan
        fpr_b  = FPb / (FPb + TNb) if (FPb+TNb)>0 else np.nan
        prec_b = TPb / (TPb + FPb) if (TPb+FPb)>0 else np.nan
        rec_b  = tpr_b
        tprs.append(tpr_b); fprs.append(fpr_b)
        precs.append(prec_b); recs.append(rec_b)

    def ci(arr):
        arr = np.asarray(arr, float)
        arr = arr[~np.isnan(arr)]
        if len(arr)==0: return (np.nan, np.nan)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return dict(
        name=expert_col, n=n,
        fpr=fpr, tpr=tpr, prec=prec, rec=rec,
        fpr_ci=ci(fprs), tpr_ci=ci(tprs),
        prec_ci=ci(precs), rec_ci=ci(recs)
    )

# ---------- Main function ----------
def plot_combined_roc_pr_with_experts(
    file_path,
    expert_cols,
    new_expert_name,
    n_boot=1000,
    out_dir="figs",
    model_true_col="true",       # 0/1, may contain NaN/None
    model_prob_col="prob"        # float probability
):
    """
    Plot 1 ROC + 1 PR curve (combined: non-other as positive class)
      - Model curve: from df[true], df[prob]
      - Expert points + cross CI: from r vs majority_without_{r} comparison
    """
    ensure_dir(out_dir)

    df=pd.read_csv(file_path)

    # -- Model: clean true/prob --
    if model_true_col not in df.columns or model_prob_col not in df.columns:
        raise ValueError("Missing true/prob columns.")

    y_true = df[model_true_col].copy()
    y_score = df[model_prob_col].copy()
    mask_model = y_true.notna() & y_score.notna()
    y_true = y_true[mask_model].astype(int).to_numpy()
    y_score = y_score[mask_model].astype(float).to_numpy()

    if len(y_true) == 0:
        raise ValueError("No samples available for model ROC/PR (true/prob contains nulls).")

    # -- Full-data ROC/PR --
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # -- Bootstrap (model curves) --
    fpr_grid = np.linspace(0, 1, 201)
    rec_grid = np.linspace(0, 1, 201)

    rng = np.random.default_rng(2025)
    tpr_curves, prec_curves = [], []
    aucs, aps = [], []

    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ys = y_score[idx]

        # Skip single-class bootstrap samples
        if yt.max() == yt.min():
            continue

        # ROC
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        aucs.append(auc(fpr_b, tpr_b))
        tpr_curves.append(np.interp(fpr_grid, fpr_b, tpr_b))

        # PR (deduplicate rec before interpolation)
        prec_b, rec_b, _ = precision_recall_curve(yt, ys)
        rec_b, uniq_idx = np.unique(rec_b, return_index=True)
        prec_b = prec_b[uniq_idx]
        prec_curves.append(np.interp(rec_grid, rec_b, prec_b))
        aps.append(average_precision_score(yt, ys))

    if len(tpr_curves)==0 or len(prec_curves)==0:
        raise ValueError("No valid curves after bootstrap; possible extreme class imbalance.")

    roc_lo, roc_med, roc_hi = _percentile_band(tpr_curves, fpr_grid)
    pr_lo,  pr_med,  pr_hi  = _percentile_band(prec_curves, rec_grid)

    # -- Expert points + CI (based on r vs majority_without_r) --
    expert_pts = []
    for r in expert_cols:
        loo_col = f"majority_without_{r}"
        if loo_col not in df.columns:
            raise ValueError(f"Missing column: {loo_col}")
        stat = _expert_point_and_ci_from_cols(df, r, loo_col, n_boot=n_boot, rng=rng)
        expert_pts.append(stat)

    colors = plt.cm.tab10.colors

    # ===== Plot ROC =====
    plt.figure(figsize=(6, 5.2))
    # Model CI shading
    plt.fill_between(fpr_grid, roc_lo, roc_hi, alpha=0.25, label="Model 95% CI (bootstrap)")
    # Full model curve
    plt.plot(fpr, tpr, lw=2, label=f"Model (AUC {roc_auc:.3f})")
    # Expert points + gray cross
    for i, p in enumerate(expert_pts):
        if np.isnan(p["fpr"]) or np.isnan(p["tpr"]):
            continue
        # Cross (gray, dashed)
        fpr_lo, fpr_hi = p["fpr_ci"]
        tpr_lo, tpr_hi = p["tpr_ci"]
        if not np.isnan(fpr_lo) and not np.isnan(fpr_hi):
            plt.plot([fpr_lo, fpr_hi], [p["tpr"], p["tpr"]],
                     color="gray", lw=1.2)
        if not np.isnan(tpr_lo) and not np.isnan(tpr_hi):
            plt.plot([p["fpr"], p["fpr"]], [tpr_lo, tpr_hi],
                     color="gray", lw=1.2)
        # Colored point
        plt.scatter(p["fpr"], p["tpr"], s=65, color=colors[i % len(colors)],
                    label=f"{new_expert_name[i]} (n={p['n']})", marker="^")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC (PD vs other)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "combined_roc.png")
    plt.savefig(roc_path, dpi=300); plt.close()

    # ===== Plot PR =====
    plt.figure(figsize=(6, 5.2))
    # Model CI shading (mask out NaN)
    mask = ~(np.isnan(pr_lo) | np.isnan(pr_hi))
    if mask.any():
        plt.fill_between(rec_grid[mask], pr_lo[mask], pr_hi[mask],
                         alpha=0.25, label="Model 95% CI (bootstrap)")
    # Full model PR curve (triangle marker optional)
    plt.plot(rec, prec, lw=2, marker="^", markersize=3,
             label=f"Model (AP {ap:.3f})")
    # Expert points + gray dashed cross (horizontal=Recall CI, vertical=Precision CI)
    for i, p in enumerate(expert_pts):
        if np.isnan(p["rec"]) or np.isnan(p["prec"]):
            continue
        rec_lo, rec_hi = p["rec_ci"]
        prec_lo, prec_hi = p["prec_ci"]
        if not np.isnan(rec_lo) and not np.isnan(rec_hi):
            plt.plot([rec_lo, rec_hi], [p["prec"], p["prec"]],
                     color="gray", lw=1.2)
        if not np.isnan(prec_lo) and not np.isnan(prec_hi):
            plt.plot([p["rec"], p["rec"]], [prec_lo, prec_hi],
                     color="gray", lw=1.2)
        plt.scatter(p["rec"], p["prec"], s=65, color=colors[i % len(colors)],
                    label=f"{new_expert_name[i]} (n={p['n']})", marker="^")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined PR (PD vs other)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, "combined_pr.png")
    plt.savefig(pr_path, dpi=300); plt.close()

    # Statistics
    auc_ci = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
    ap_ci  = (np.percentile(aps, 2.5),  np.percentile(aps, 97.5))

    print(f"[Saved] {roc_path} | AUC={roc_auc:.4f} 95%CI[{auc_ci[0]:.4f},{auc_ci[1]:.4f}]")
    print(f"[Saved] {pr_path}  |  AP={ap:.4f}   95%CI[{ap_ci[0]:.4f},{ap_ci[1]:.4f}]")

    return {
        "n_model": int(n),
        "auc": float(roc_auc), "auc_ci": (float(auc_ci[0]), float(auc_ci[1])),
        "ap": float(ap), "ap_ci": (float(ap_ci[0]), float(ap_ci[1])),
        "roc_path": roc_path, "pr_path": pr_path,
        "expert_points": expert_pts
    }




def calibration_curve_points(y_true, y_prob, n_bins=10, strategy="quantile"):
    """
    Returns (bin_conf, bin_acc)
    bin_conf: mean predicted probability per bin
    bin_acc : actual hit rate per bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins+1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0, 1, n_bins+1)  # fallback
    else:
        edges = np.linspace(0, 1, n_bins+1)

    idx = np.digitize(y_prob, edges[1:-1], right=False)

    bin_conf, bin_acc = [], []
    for b in range(len(edges)-1):
        m = (idx == b)
        if m.sum() == 0:
            continue
        bin_conf.append(y_prob[m].mean())
        bin_acc.append(y_true[m].mean())
    return np.array(bin_conf), np.array(bin_acc)

def plot_calibration_curve(file_path , true_col="true", prob_col="prob",
                           n_bins=10, strategy="quantile",
                           out_path="figs/calibration.png"):
    # Valid samples
    df=pd.read_csv(file_path)
    data = df[[true_col, prob_col]].dropna().copy()
    y_true = data[true_col].astype(float).to_numpy()
    y_prob = data[prob_col].astype(float).to_numpy()

    if len(y_true) == 0:
        raise ValueError("No valid samples")

    # Compute binning curve
    bin_conf, bin_acc = calibration_curve_points(y_true, y_prob,
                                                 n_bins=n_bins,
                                                 strategy=strategy)

    # Plot
    ensure_dir(out_path)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="Perfect calibration")
    plt.plot(bin_conf, bin_acc, "o-", lw=2, label="Model")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path






# # ========= Utilities =========
# def _to_list_col(s):
#     """Safely convert strings like '[]' or '[1,2]' to list; return as-is if already a list."""
#     if isinstance(s, list):
#         return s
#     if pd.isna(s):
#         return []
#     s = str(s).strip()
#     try:
#         v = ast.literal_eval(s)
#         return v if isinstance(v, list) else [v]
#     except Exception:
#         # Fallback: split by comma/space and try to convert to int
#         parts = [x for x in re.split(r'[\s,]+', s) if x!='']
#         out = []
#         for x in parts:
#             try: out.append(int(x))
#             except: out.append(x)
#         return out
#
# def _get_prob_matrix(df, classes):
#     cols = [f"class_{int(c)}_prob" for c in classes]
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing probability columns: {missing}")
#     return df[cols].to_numpy(dtype=float), cols
#
# def _infer_classes_from_probs(df):
#     return sorted([int(m.group(1)) for c in df.columns if (m:=re.match(r"^class_(\d+)_prob$", c))])
#
# # ========= A. One-vs-Rest: plot ROC & PR =========
# def plot_multilabel_ovr_roc_pr(
#     df,
#     true_col="true_labels",
#     class_order=None,
#     out_prefix="ovr"
# ):
#     # 1) Parse true labels
#     y_true_lists = df[true_col].apply(_to_list_col).tolist()
#
#     # 2) Class order
#     if class_order is None:
#         class_order = _infer_classes_from_probs(df)
#         if not class_order:
#             # Infer from true label set
#             class_order = sorted({lbl for row in y_true_lists for lbl in row})
#
#     # 3) one-hot true labels & probability matrix
#     mlb = MultiLabelBinarizer(classes=class_order)
#     Y_true = mlb.fit_transform(y_true_lists)              # [N, C]
#     Y_score, prob_cols = _get_prob_matrix(df, class_order)  # [N, C]
#
#     classes = list(mlb.classes_)
#     C = len(classes)
#
#     # 4) Per-class ROC/PR
#     fpr, tpr, roc_auc = {}, {}, {}
#     prec, rec, ap = {}, {}, {}
#     for i, cls in enumerate(classes):
#         # Skip class if no positive/negative samples (set NaN)
#         if Y_true[:, i].max() == Y_true[:, i].min():
#             fpr[cls], tpr[cls], roc_auc[cls] = np.array([0,1]), np.array([0,1]), np.nan
#             prec[cls], rec[cls], ap[cls] = np.array([1,1]), np.array([0,1]), np.nan
#             continue
#         fpr[cls], tpr[cls], _ = roc_curve(Y_true[:, i], Y_score[:, i])
#         roc_auc[cls] = auc(fpr[cls], tpr[cls])
#         prec[cls], rec[cls], _ = precision_recall_curve(Y_true[:, i], Y_score[:, i])
#         ap[cls] = average_precision_score(Y_true[:, i], Y_score[:, i])
#
#     # 5) micro/macro
#     # micro
#     y_flat = Y_true.ravel()
#     s_flat = Y_score.ravel()
#     micro_auc = macro_auc = micro_ap = macro_ap = None
#     if y_flat.max() != y_flat.min():
#         fpr_micro, tpr_micro, _ = roc_curve(y_flat, s_flat)
#         micro_auc = auc(fpr_micro, tpr_micro)
#         prec_micro, rec_micro, _ = precision_recall_curve(y_flat, s_flat)
#         micro_ap = average_precision_score(y_flat, s_flat)
#     # macro
#     valid_roc_aucs = [v for v in roc_auc.values() if not np.isnan(v)]
#     valid_aps = [v for v in ap.values() if not np.isnan(v)]
#     if valid_roc_aucs:
#         macro_auc = float(np.mean(valid_roc_aucs))
#     if valid_aps:
#         macro_ap = float(np.mean(valid_aps))
#
#     # 6) Plot ROC
#     plt.figure(figsize=(7,6))
#     for i, cls in enumerate(classes):
#         if np.isnan(roc_auc[cls]):  # Skip classes that cannot be computed
#             continue
#         plt.plot(fpr[cls], tpr[cls], lw=1.6, label=f"class {cls}: AUC={roc_auc[cls]:.3f}")
#     if micro_auc is not None:
#         plt.plot(fpr_micro, tpr_micro, lw=2.2, label=f"micro avg: AUC={micro_auc:.3f}")
#     plt.plot([0,1],[0,1],'k--',lw=1)
#     ttl_tail = []
#     if micro_auc is not None: ttl_tail.append(f"micro={micro_auc:.3f}")
#     if macro_auc is not None: ttl_tail.append(f"macro={macro_auc:.3f}")
#     ttl = "OvR ROC" + ("" if not ttl_tail else f"  [{' | '.join(ttl_tail)}]")
#     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
#     plt.title(ttl)
#     plt.legend(loc="lower right", fontsize=9, ncol=2)
#     plt.tight_layout()
#     roc_path = f"{out_prefix}_roc.png"
#     plt.savefig(roc_path, dpi=300); plt.close()
#
#     # 7) Plot PR
#     plt.figure(figsize=(7,6))
#     for i, cls in enumerate(classes):
#         if np.isnan(ap[cls]):  # Skip classes that cannot be computed
#             continue
#         plt.plot(rec[cls], prec[cls], lw=1.6, label=f"class {cls}: AP={ap[cls]:.3f}")
#     if micro_ap is not None:
#         plt.plot(rec_micro, prec_micro, lw=2.2, label=f"micro avg: AP={micro_ap:.3f}")
#     plt.xlim(0,1); plt.ylim(0,1)
#     ttl_tail = []
#     if micro_ap is not None: ttl_tail.append(f"micro={micro_ap:.3f}")
#     if macro_ap is not None: ttl_tail.append(f"macro={macro_ap:.3f}")
#     ttl = "OvR PR" + ("" if not ttl_tail else f"  [{' | '.join(ttl_tail)}]")
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(ttl)
#     plt.legend(loc="lower left", fontsize=9, ncol=2)
#     plt.tight_layout()
#     pr_path = f"{out_prefix}_pr.png"
#     plt.savefig(pr_path, dpi=300); plt.close()
#
#     return {
#         "classes": classes, "per_class_auc": roc_auc, "per_class_ap": ap,
#         "micro_auc": micro_auc, "macro_auc": macro_auc,
#         "micro_ap": micro_ap, "macro_ap": macro_ap,
#         "roc_path": roc_path, "pr_path": pr_path
#     }

# # ========= B. Exact-Match: Top-K prediction + confidence score, plot ROC & PR =========
# def plot_exactmatch_roc_pr(
#     df,
#     true_col="true_labels",
#     class_order=None,
#     out_prefix="exact"
# ):
#     # 1) Parse true labels
#     y_true_lists = df[true_col].apply(_to_list_col).tolist()
#
#     # 2) Class order
#     if class_order is None:
#         class_order = _infer_classes_from_probs(df)
#         if not class_order:
#             class_order = sorted({lbl for row in y_true_lists for lbl in row})
#
#     # 3) Probability matrix & one-hot true labels
#     Y_score, prob_cols = _get_prob_matrix(df, class_order)  # [N, C]
#     mlb = MultiLabelBinarizer(classes=class_order)
#     Y_true = mlb.fit_transform(y_true_lists)                # [N, C]
#
#     N, C = Y_score.shape
#     y_true_exact = np.zeros(N, dtype=int)
#     y_score_exact = np.zeros(N, dtype=float)
#
#     for i in range(N):
#         probs = Y_score[i]
#         true_vec = Y_true[i]
#         true_idx = np.where(true_vec == 1)[0]
#         k = len(true_idx)
#
#         if k == 0:
#             # True set is empty: Top-0=empty set; exact=1 iff we also predict empty set
#             # Use threshold-free definition: if max prob < 0.5, "lean toward empty set"
#             pred_idx = np.array([], dtype=int)
#             exact = (len(pred_idx) == 0)
#             min_true = 1.0                     # For empty set, define min_true=1 (no penalty)
#             max_false = float(np.max(probs))   # all "negative" classes
#         else:
#             # Top-K prediction set
#             pred_idx = np.argsort(-probs)[:k]
#             exact = int(set(pred_idx) == set(true_idx))
#             min_true = float(np.min(probs[true_idx]))
#             max_false = float(np.max(probs[[j for j in range(C) if j not in true_idx]])) if k < C else 0.0
#
#         y_true_exact[i] = exact
#         # Confidence: larger true, smaller false is better
#         y_score_exact[i] = float(np.clip(min_true * (1.0 - max_false), 0.0, 1.0))
#
#     # If all samples have the same label (all 0 or all 1), ROC/PR cannot be computed
#     if y_true_exact.max() == y_true_exact.min():
#         raise ValueError("Exact-match labels are all equal (all positive or all negative); cannot compute ROC/PR.")
#
#     # ROC
#     fpr, tpr, _ = roc_curve(y_true_exact, y_score_exact)
#     auc_val = auc(fpr, tpr)
#
#     # PR
#     prec, rec, _ = precision_recall_curve(y_true_exact, y_score_exact)
#     ap_val = average_precision_score(y_true_exact, y_score_exact)
#
#     # Plot
#     # ROC
#     plt.figure(figsize=(6.2,5.2))
#     plt.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
#     plt.plot([0,1],[0,1],'k--',lw=1)
#     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
#     plt.title("Exact-Match ROC (Top-K predicted set)")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     roc_path = f"{out_prefix}_roc.png"
#     plt.savefig(roc_path, dpi=300); plt.close()
#
#     # PR
#     plt.figure(figsize=(6.2,5.2))
#     plt.plot(rec, prec, lw=2, label=f"AP={ap_val:.3f}")
#     plt.xlim(0,1); plt.ylim(0,1)
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title("Exact-Match PR (Top-K predicted set)")
#     plt.legend(loc="lower left")
#     plt.tight_layout()
#     pr_path = f"{out_prefix}_pr.png"
#     plt.savefig(pr_path, dpi=300); plt.close()
#
#     return {
#         "roc_path": roc_path, "pr_path": pr_path,
#         "auc": float(auc_val), "ap": float(ap_val)
#     }


import re, ast, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ---------- General utilities ----------
def parse_list_cell(x):
    """Safely convert '[]' / '[1,2]' / '1,2' etc. in a DataFrame to list."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        parts = [t for t in re.split(r'[\s,]+', s) if t != '']
        out = []
        for t in parts:
            try: out.append(int(t))
            except: out.append(t)
        return out

def infer_classes_from_prob_cols(df):
    """Infer sorted list of class indices from class_<k>_prob column names."""
    ks = []
    for c in df.columns:
        m = re.match(r"^class_(\d+)_prob$", c)
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)

def get_prob_matrix(df, classes):
    cols = [f"class_{int(k)}_prob" for k in classes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    P = df[cols].to_numpy(dtype=float)
    # Clip extreme values to avoid log(0)
    eps = 1e-12
    return np.clip(P, eps, 1 - eps), cols

# ---------- A) OvR: per-class ROC+PR + micro/macro ----------
def plot_multilabel_ovr_roc_pr(df, true_col="true_labels", out_prefix="ovr"):
    true_lists = df[true_col].apply(parse_list_cell).tolist()
    classes = infer_classes_from_prob_cols(df)
    if not classes:
        # Fallback: infer from true labels
        classes = sorted({lbl for row in true_lists for lbl in row})

    mlb = MultiLabelBinarizer(classes=classes)
    Y_true = mlb.fit_transform(true_lists)              # [N,C]
    P, prob_cols = get_prob_matrix(df, classes)         # [N,C]

    # Per-class ROC/PR
    fpr, tpr, roc_auc = {}, {}, {}
    prec, rec, ap = {}, {}, {}
    for i, cls in enumerate(classes):
        yi = Y_true[:, i]
        pi = P[:, i]
        if yi.max() == yi.min():  # all 0 or all 1, cannot plot
            roc_auc[cls] = np.nan; ap[cls] = np.nan
            continue
        fpr[cls], tpr[cls], _ = roc_curve(yi, pi)
        roc_auc[cls] = auc(fpr[cls], tpr[cls])
        prec[cls], rec[cls], _ = precision_recall_curve(yi, pi)
        ap[cls] = average_precision_score(yi, pi)

    # micro
    yf = Y_true.ravel(); pf = P.ravel()
    fpr_micro = tpr_micro = rec_micro = prec_micro = None
    micro_auc = micro_ap = None
    if yf.max() != yf.min():
        fpr_micro, tpr_micro, _ = roc_curve(yf, pf)
        micro_auc = auc(fpr_micro, tpr_micro)
        prec_micro, rec_micro, _ = precision_recall_curve(yf, pf)
        micro_ap = average_precision_score(yf, pf)

    # macro
    valid_aucs = [v for v in roc_auc.values() if not np.isnan(v)]
    valid_aps  = [v for v in ap.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
    macro_ap  = float(np.mean(valid_aps))  if valid_aps  else None

    # Plot ROC
    plt.figure(figsize=(7,6))
    for cls in classes:
        if cls in roc_auc and not np.isnan(roc_auc[cls]):
            plt.plot(fpr[cls], tpr[cls], lw=1.6, label=f"channel {cls}: AUC={roc_auc[cls]:.3f}")
    if micro_auc is not None:
        plt.plot(fpr_micro, tpr_micro, lw=2.2, label=f"micro avg: AUC={micro_auc:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1)
    tail = []
    if micro_auc is not None: tail.append(f"micro={micro_auc:.3f}")
    if macro_auc is not None: tail.append(f"macro={macro_auc:.3f}")
    plt.title("OvR ROC" + ("" if not tail else f"  [{' | '.join(tail)}]"))
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout()
    roc_path = f"{out_prefix}_roc.png"; plt.savefig(roc_path, dpi=300); plt.close()

    # Plot PR
    plt.figure(figsize=(7,6))
    for cls in classes:
        if cls in ap and not np.isnan(ap[cls]):
            plt.plot(rec[cls], prec[cls], lw=1.6, label=f"channel {cls}: AP={ap[cls]:.3f}")
    if micro_ap is not None:
        plt.plot(rec_micro, prec_micro, lw=2.2, label=f"micro avg: AP={micro_ap:.3f}")
    tail = []
    if micro_ap is not None: tail.append(f"micro={micro_ap:.3f}")
    if macro_ap is not None: tail.append(f"macro={macro_ap:.3f}")
    plt.title("OvR PR" + ("" if not tail else f"  [{' | '.join(tail)}]"))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.legend(loc="lower left", fontsize=9, ncol=2)
    plt.tight_layout()
    pr_path = f"{out_prefix}_pr.png"; plt.savefig(pr_path, dpi=300); plt.close()

    return {"roc_path": roc_path, "pr_path": pr_path,
            "per_class_auc": roc_auc, "per_class_ap": ap,
            "micro_auc": micro_auc, "macro_auc": macro_auc,
            "micro_ap": micro_ap, "macro_ap": macro_ap}

# ---------- B) Exact-match: set fully matches -> ROC+PR ----------
def plot_exactmatch_roc_pr(df, true_col="true_labels", pred_col="pred_labels", out_prefix="exact"):
    true_lists = df[true_col].apply(parse_list_cell).tolist()
    pred_lists = df[pred_col].apply(parse_list_cell).tolist()

    classes = infer_classes_from_prob_cols(df)
    if not classes:
        # No probability columns means no continuous scoring for exact-match
        raise ValueError("No class_n_prob probability columns found.")

    P, _ = get_prob_matrix(df, classes)   # [N,C]

    N, C = P.shape
    y_true_exact = np.zeros(N, dtype=int)
    score_loglik = np.zeros(N, dtype=float)

    for i in range(N):
        T = set(true_lists[i])  # True set
        H = set(pred_lists[i])  # Predicted set (discrete)
        y_true_exact[i] = int(T == H)

        # Use independent Bernoulli log-likelihood of "predicted set H is true" as continuous score:
        # s = sum_{k in H} log p_k + sum_{k not in H} log(1-p_k)
        mask = np.zeros(C, dtype=bool)
        # Map class indices to column indices: assume class numbers match column order (k in class_k_prob)
        # If not numbered 0..C-1 consecutively, build an index mapping:
        idx_of = {c:i for i,c in enumerate(classes)}
        for lab in H:
            if lab in idx_of:
                mask[idx_of[lab]] = True

        s = np.sum(np.log(P[i, mask])) + np.sum(np.log(1.0 - P[i, ~mask]))
        score_loglik[i] = s

    # Monotonically normalize to [0,1] (preserves ROC/PR ranking properties)
    s_min, s_max = np.min(score_loglik), np.max(score_loglik)
    if math.isclose(s_max, s_min):
        y_score = np.ones_like(score_loglik) * 0.5
    else:
        y_score = (score_loglik - s_min) / (s_max - s_min)

    # If y_true_exact is all 0 or all 1, ROC/PR cannot be computed
    if y_true_exact.max() == y_true_exact.min():
        raise ValueError("Exact-match labels are all equal (all positive or all negative); cannot compute ROC/PR.")

    # ROC
    fpr, tpr, _ = roc_curve(y_true_exact, y_score)
    auc_val = auc(fpr, tpr)

    # PR
    prec, rec, _ = precision_recall_curve(y_true_exact, y_score)
    ap_val = average_precision_score(y_true_exact, y_score)

    # Plot
    # ROC
    plt.figure(figsize=(6.2,5.2))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Exact-match ROC (set equality)")
    plt.legend(loc="lower right"); plt.tight_layout()
    roc_path = f"{out_prefix}_roc.png"; plt.savefig(roc_path, dpi=300); plt.close()

    # PR
    plt.figure(figsize=(6.2,5.2))
    plt.plot(rec, prec, lw=2, label=f"AP={ap_val:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.title("Exact-match PR (set equality)")
    plt.legend(loc="lower left"); plt.tight_layout()
    pr_path = f"{out_prefix}_pr.png"; plt.savefig(pr_path, dpi=300); plt.close()

    return {"roc_path": roc_path, "pr_path": pr_path,
            "auc": float(auc_val), "ap": float(ap_val)}



def plot_roc_pr_from_csv(
    csv_path: str,
    expert_cols: list,
    prob_col: str,
    out_dir: str,
    roc_name: str = "roc.png",
    pr_name: str = "pr.png",
    pos_label=1,
    title_prefix: str = ""
):
    """
    Read csv_path:
      - Use majority vote of expert_cols as ground truth (skip ties or missing)
      - Use prob_col as model prediction score
      - Plot ROC and PR curves and save to out_dir
    Returns:
      dict( n_total, n_used, n_skipped, auc_roc, ap_pr )
    """
    # Read
    df = pd.read_csv(csv_path)

    # Keep only required columns
    cols_needed = expert_cols + [prob_col]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Compute majority vote ground truth (skip ties/missing)
    y_true, y_score = [], []
    for idx, row in df[cols_needed].iterrows():
        experts = row[expert_cols]
        # Drop missing expert values
        ex_valid = experts.dropna()
        if ex_valid.empty:
            continue
        # Majority vote
        cnt = Counter(ex_valid)
        common = cnt.most_common()
        # Skip ties
        if len(common) > 1 and common[0][1] == common[1][1]:
            continue

        maj = common[0][0]
        # Skip missing prob
        p = row[prob_col]
        if pd.isna(p):
            continue

        y_true.append(1 if maj == pos_label else 0)
        y_score.append(float(p))

    n_total = len(df)
    n_used  = len(y_true)
    n_skip  = n_total - n_used

    if n_used == 0 or len(set(y_true)) < 2:
        raise ValueError("Insufficient valid samples or labels do not form a binary classification; cannot plot ROC/PR.")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # ===== ROC =====
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc_roc = auc(fpr, tpr)

    # ===== PR =====
    prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    ap_pr = average_precision_score(y_true, y_score, pos_label=1)

    # ===== Save figures =====
    os.makedirs(out_dir, exist_ok=True)
    dpi = 300

    # ROC
    plt.figure(figsize=(5.4, 4.4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_roc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1, alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title((title_prefix + " ROC").strip())
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, roc_name)
    plt.savefig(roc_path, dpi=dpi)
    plt.close()

    # PR
    plt.figure(figsize=(5.4, 4.4))
    plt.plot(rec, prec, lw=2, label=f"AP = {ap_pr:.3f}")
    # Reference line: positive class baseline
    base = (y_true == 1).mean()
    plt.hlines(base, 0, 1, linestyles="--", linewidth=1, alpha=0.6)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title((title_prefix + " PR").strip())
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, pr_name)
    plt.savefig(pr_path, dpi=dpi)
    plt.close()



if __name__ == "__main__":


    # PD ####################################################################################
    # result_path='/data/partial_PD/results/model_experts_LPD_lateralize_scores_20250827.csv'
    # label_cols = ["mbw_label", "wyk_label", "yyy_label", "morgoth_pred"]
    #
    # new_label_cols_name= ["rater 1", "rater 2", "rater 3","morgoth_pred"]
    #new_label_cols_name=label_cols

    # combine_experts_model_results_pd(
    #     pd_train_test_list_path='/data/partial_PD/PD_train_test_list.csv',
    #     expert_results_path='/data/partial_PD/LPD_lateralize_scores_20250826.xlsx',
    #     model_results_dir='test_data/PD_pred_1s',
    #     expert_cols=label_cols[:-1],
    #     out_path=result_path,
    #
    # )

    # results = compute_irr(file_path=result_path, cols=label_cols,new_label_cols_name=new_label_cols_name)
    #
    # plot_pairwise_heatmap(results, new_label_cols_name,
    #                       out_path="/data/partial_PD/results/irr_pairwise_heatmap.png")
    #
    #
    # plot_vs_majority_squares(results,
    #                          out_path="/data/partial_PD/results/irr_vs_majority_squares.png")

    # labels=['right','left','both']
    # prob_cols=['r_prob','l_prob','b_prob']
    # for i in range(len(labels)):
    #     plot_roc_pr_with_bootstrap(
    #             label=labels[i],
    #             file_path = result_path,
    #             expert_cols=label_cols[:-1],
    #             new_label_cols_name=new_label_cols_name[:-1],
    #             prob_col = prob_cols[i],
    #             n_boot = 1000,
    #             out_dir= "/data/partial_PD/results"
    #     )

    # plot_combined_roc_pr_with_experts(
    #         file_path = result_path,
    #         expert_cols=label_cols[:-1],
    #         new_expert_name=new_label_cols_name[:-1],
    #         n_boot=1000,
    #         out_dir="/data/partial_PD/results",
    #         model_true_col="true",
    #         model_prob_col="prob"
    # )

    # plot_calibration_curve(file_path = result_path,
    #                        true_col="true",
    #                        prob_col="prob",
    #                        n_bins=10,
    #                        strategy="quantile",
    #                        out_path="/data/partial_PD/results/combined_calibration.png")

    # # VW ####################################################################################
    # result_path = '/data/VW/results/model_experts_VW_scores_20250829.csv'
    # label_cols = ["mbw_label", "fab_label", "tyz_label", "wyk_label", "yyy_label", "morgoth_pred"]
    # new_label_cols_name = ["Expert 1", "Expert 2", "Expert 3", "Expert 4", "Expert 5", "morgoth_pred"]
    #
    # # combine_experts_model_results_vw(
    # #     expert_results_path='/data/VW/vw_scores_20250828.xlsx',
    # #     model_results_dir='/data/VW/VW_pred_8p',
    # #     result_stride=8,
    # #     vw_label_path='/data/VW/list_vw_events.csv',
    # #     out_path=result_path)
    #
    #
    # results = compute_irr(file_path=result_path, cols=label_cols,
    # #new_label_cols_name=new_label_cols_name
    # )
    #
    # plot_pairwise_heatmap(results=results,cols= label_cols,
    #                       new_label_cols_name=new_label_cols_name,
    #                       out_path="/data/VW/results/irr_pairwise_heatmap.png")
    #
    #
    # plot_vs_majority_squares(results,
    #                          expert_cols=label_cols[:-1],
    #                          new_label_cols_name=new_label_cols_name[:-1],
    #                          out_path="/data/VW/results/irr_vs_majority_squares.png")
    #
    #
    # plot_roc_pr_with_bootstrap(
    #         label="vw",
    #         file_path = result_path,
    #         expert_cols=label_cols[:-1],
    #         new_label_cols_name=new_label_cols_name[:-1],
    #         prob_col = "morgoth_prob",
    #         n_boot = 1000,
    #         out_dir= "/data/VW/results"
    # )

    # BIRD ####################################################################################

    result_path = '/data/BIRD/BIRD_experts_model_results.csv'
    label_cols = ["score_mbw", "score_yyy", "pred"]
    new_label_cols_name = ["Expert 1", "Expert 2", "morgoth_pred"]

    results = compute_irr(file_path=result_path,
                          cols=label_cols,
                          thresholds={'pred':0.5},
                          # new_label_cols_name=new_label_cols_name
                          )

    plot_pairwise_heatmap(results=results, cols=label_cols,
                          new_label_cols_name=new_label_cols_name,
                          out_path="/data/BIRD/results/irr_pairwise_heatmap.png")

    plot_vs_majority_squares(results,
                             expert_cols=label_cols[:-1],
                             new_label_cols_name=new_label_cols_name[:-1],
                             out_path="/data/BIRD/results/irr_vs_majority_squares.png")


    plot_roc_pr_from_csv(
        csv_path=result_path,
    expert_cols=label_cols[:-1],
    prob_col=label_cols[-1],
    out_dir="/data/BIRD/results/",
    roc_name = "roc.png",
    pr_name = "pr.png",
    pos_label = 1,
    title_prefix = "BIRD Detection")


# BIOD ####################################################################################

    result_path = '/data/BIPD/BIPD_experts_model_results.csv'
    label_cols = ["score_mbw", "score_yyy", "pred"]
    new_label_cols_name = ["Expert 1", "Expert 2", "morgoth_pred"]

    results = compute_irr(file_path=result_path,
                          cols=label_cols,
                          thresholds={'pred':0.5},
                          # new_label_cols_name=new_label_cols_name
                          )

    plot_pairwise_heatmap(results=results, cols=label_cols,
                          new_label_cols_name=new_label_cols_name,
                          out_path="/data/BIPD/results/irr_pairwise_heatmap.png")

    plot_vs_majority_squares(results,
                             expert_cols=label_cols[:-1],
                             new_label_cols_name=new_label_cols_name[:-1],
                             out_path="/data/BIPD/results/irr_vs_majority_squares.png")


   # spike localization ####################################################################################

    # df=pd.read_csv('/data/SPIKE_localization/processed_1second/test/pred.csv')
    # os.makedirs('/data/SPIKE_localization/results', exist_ok=True)

    # # 1) One-vs-Rest: per-class ROC/PR + micro/macro
    # res_ovr = plot_multilabel_ovr_roc_pr(
    #     df,
    #     true_col="true_labels",
    #     class_order=None,  # automatically infer from class_*.prob columns
    #     out_prefix="/data/SPIKE_localization/results/ovr"
    # )
    #
    # # 2) Exact-Match: compare Top-K predicted set vs true labels, plot ROC/PR (1=exact match)
    # res_exact = plot_exactmatch_roc_pr(
    #     df,
    #     true_col="true_labels",
    #     class_order=None,
    #     out_prefix="/data/SPIKE_localization/results/exact"
    # )

    # Your DataFrame: df (with true_labels, pred_labels, and class_n_prob columns)

    #1) OvR
    # res_ovr = plot_multilabel_ovr_roc_pr(df, true_col="true_labels", out_prefix="/data/SPIKE_localization/results/ovr")
    # print(res_ovr["roc_path"], res_ovr["pr_path"])
    #
    # # 2) Exact-match (set equality)
    # res_exact = plot_exactmatch_roc_pr(df, true_col="true_labels", pred_col="pred_labels", out_prefix="/data/SPIKE_localization/results/exact")
    # print(res_exact["roc_path"], res_exact["pr_path"])



    pass

# sudo ~/miniconda3/envs/torchenv/bin/python model_experts_irr.py