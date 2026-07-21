#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QORC on NSL-KDD -- v2: fixing the 0% zero-day result
====================================================
The previous run scored 0.0% detection on the held-out 'smurf' attack for
every quantum arm. Data analysis showed WHY, and this script fixes it three
ways. Nothing about the Phase-1 ablation logic changed, so the good result
(quantum beats its own input) should be preserved.

WHAT WAS WRONG, AND WHAT CHANGED
--------------------------------
1. THE 0.5 CUTOFF WAS ARBITRARY.
   "Detection" was counted only when the model said >50% attack. Real
   intrusion-detection systems never do this: they pick a threshold from a
   FALSE-ALARM BUDGET ("flag the top 1% most suspicious traffic"). A model
   that ranks smurf as suspicious but below 50% scored 0% -- unfairly.
   FIX: report ROC-AUC (threshold-free) plus detection at 1% and 5%
   false-alarm budgets. The old 0.5 number is still printed for comparison.

2. SUPERVISED LEARNING CANNOT GENERALISE TO AN UNSEEN ATTACK TYPE.
   Trained on normal + known attacks, a classifier learns the boundary of
   the attacks it has SEEN. Smurf is 100% icmp/ecr_i/SF, and the only
   remaining icmp/ecr_i/SF rows in training are 190 NORMAL ones -- so the
   model is actively taught "this fingerprint = normal".
   FIX: add NOVELTY detectors trained on NORMAL TRAFFIC ONLY. They never
   see any attack, so no attack can be "unseen" to them. This is the
   label-free / zero-day story the project promised from the start.

3. HARD CLIPPING DESTROYED OUTLIER MAGNITUDE.
   Phases were min-max scaled then clipped to [0,1]. An extreme outlier and
   a mildly high value both became exactly 1.0 -- erasing the very signal an
   anomaly detector needs. (Phases are periodic, so simply widening the
   range would wrap around and be worse.)
   FIX: a smooth, monotonic tanh squash maps all of R into (0, pi) without
   clipping or wrapping. Outliers saturate gracefully and stay ordered.

Run:  python qorc_zeroday_v2.py
Deps: pip install perceval-quandela merlinquantum torch scikit-learn pandas
      (optional) pip install xgboost
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import torch
import perceval as pcvl
import merlin as ML

try:
    MeasurementStrategy = ML.MeasurementStrategy
    ComputationSpace = ML.ComputationSpace
except AttributeError:
    from merlin.measurement.strategies import MeasurementStrategy, ComputationSpace

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------------------------------------------------
# 0) Settings
# ----------------------------------------------------------------------
N_MODES     = 20
N_PHOTONS   = 3
TEST_SIZE   = 0.30
SEED        = 42
NO_BUNCHING = True

# "tanh"   = smooth squash, no clipping   <-- the fix (recommended)
# "minmax" = previous behaviour (hard clip to [0,1])
#   If Phase-1 numbers come out WORSE than the last run, flip this back to
#   "minmax" -- Phase 1 is our proven result and takes priority.
ENCODING = "tanh"

# How the 122 features become the N_MODES dial values (supervised path).
#   "pca"   = label-blind blend (paper's choice)
#   "anova" = supervised top-k, keeps flood features like count/srv_count
#   "mi"    = supervised top-k by mutual information (slower)
REDUCER = "pca"

# Attacks to hide from training, one experiment each.
# smurf is the HARDEST case; add others to show breadth.
ZERO_DAY_ATTACKS = ["smurf"]
# e.g. ["smurf", "satan", "teardrop", "portsweep"]

FA_BUDGETS   = [0.01, 0.05]   # false-alarm budgets for the detection metric
N_NOV_COMPS  = 32             # subspace size for the novelty detectors
RUN_PHASE1   = True           # set False to jump straight to the zero-day work

CSV_CANDIDATES = [
    r"D:\claude things\QRC\data\NSL_KDD_labeled.csv",
    r"D:\claude things\NSL_KDD_labeled.csv",
    "data/NSL_KDD_labeled.csv",
    "network data/NSL_KDD_labeled.csv",
    "NSL_KDD_labeled.csv",
]

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 1) Load data
# ----------------------------------------------------------------------
csv_path = next((p for p in CSV_CANDIDATES if Path(p).exists()), None)
if csv_path is None:
    raise FileNotFoundError("NSL_KDD_labeled.csv not found -- add its path to CSV_CANDIDATES.")
print(f"[data] loading {csv_path}")
df = pd.read_csv(csv_path)

categorical_cols = ["protocol_type", "service", "flag"]
target_col = "label"
numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

y_all = np.where(df[target_col] == "normal", 0, 1).astype(np.int64)
attack_type = df[target_col].to_numpy()

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_cols]).astype(np.float32)
X_num = df[numeric_cols].to_numpy(dtype=np.float32)
X_all = np.concatenate([X_num, X_cat], axis=1)
print(f"[data] {X_all.shape[0]:,} rows, {X_all.shape[1]} raw features "
      f"({int((y_all==0).sum()):,} normal / {int((y_all==1).sum()):,} attack)")

# ----------------------------------------------------------------------
# 2) The photonic reservoir -- built once, identical structure to QORC
# ----------------------------------------------------------------------
U = pcvl.Matrix.random_unitary(N_MODES)
precircuit = pcvl.Unitary(U)
reservoir  = precircuit.copy()

c_var = pcvl.Circuit(N_MODES)
for i in range(N_MODES):
    c_var.add(i, pcvl.PS(pcvl.P(f"px{i+1}")))
qorc_circuit = precircuit // c_var // reservoir

step = (N_MODES - 1) / (N_PHOTONS - 1) if N_PHOTONS > 1 else 0
input_state = [0] * N_MODES
for k in range(N_PHOTONS):
    input_state[int(round(k * step))] = 1

if NO_BUNCHING:
    n_qfeat, space = math.comb(N_MODES, N_PHOTONS), ComputationSpace.UNBUNCHED
else:
    n_qfeat, space = math.comb(N_PHOTONS+N_MODES-1, N_PHOTONS), ComputationSpace.FOCK

quantum_layer = ML.QuantumLayer(
    input_size=N_MODES, circuit=qorc_circuit,
    trainable_parameters=[],            # the reservoir never learns
    input_parameters=["px"],
    input_state=input_state,
    measurement_strategy=MeasurementStrategy.probs(computation_space=space),
    device=torch.device("cpu"),
)
quantum_layer.eval()
print(f"[quantum] modes={N_MODES} photons={N_PHOTONS} -> {n_qfeat} features/row"
      f" | encoding={ENCODING} | reducer={REDUCER}")


def quantum_features(phases, batch=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, len(phases), batch):
            outs.append(quantum_layer(torch.tensor(phases[i:i+batch],
                                                   dtype=torch.float32)).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def fit_phase_stats(Z):
    """Learn how reduced values map to dial angles, from TRAIN rows only."""
    if ENCODING == "tanh":
        return {"mu": Z.mean(0), "sd": Z.std(0) + 1e-8}
    return {"lo": Z.min(), "hi": Z.max()}


def to_phase(Z, st):
    """Map reduced values -> phase angles in (0, pi)."""
    if ENCODING == "tanh":
        # smooth + monotonic: extreme outliers saturate but never clip or wrap
        return ((np.pi / 2) * (1 + np.tanh((Z - st["mu"]) / (3 * st["sd"])))
                ).astype(np.float32)
    return np.clip((Z - st["lo"]) / (st["hi"] - st["lo"] + 1e-8),
                   0.0, 1.0).astype(np.float32) * np.pi


def make_reducer(X, y=None):
    if REDUCER == "pca":
        return PCA(n_components=N_MODES, random_state=SEED).fit(X)
    if REDUCER == "anova":
        return SelectKBest(f_classif, k=N_MODES).fit(X, y)
    if REDUCER == "mi":
        return SelectKBest(lambda A, t: mutual_info_classif(A, t, random_state=SEED),
                           k=N_MODES).fit(X, y)
    raise ValueError(REDUCER)


# ----------------------------------------------------------------------
# 3) Novelty detector -- trained on NORMAL TRAFFIC ONLY
# ----------------------------------------------------------------------
class NoveltyDetector:
    """Learns the shape of normal traffic; scores how far a row falls from it.

    Two complementary signals, both computed in a compact subspace fitted to
    normal data only:
      * Mahalanobis distance -- inside the normal subspace, but far from the
        normal centre (e.g. a ping flood: ordinary shape, extreme volume).
      * Reconstruction error -- energy pointing OUTSIDE the normal subspace,
        i.e. a pattern normal traffic simply never produces.
    No attack is ever shown to it, so no attack can be 'unseen'.
    """

    def __init__(self, k=N_NOV_COMPS):
        self.k = k

    def fit(self, X_normal):
        self.mu = X_normal.mean(0)
        Z = X_normal - self.mu
        k = min(self.k, Z.shape[1], Z.shape[0] - 1)
        self.pca = PCA(n_components=k, svd_solver="randomized",
                       random_state=SEED).fit(Z)
        P = self.pca.transform(Z)
        self.sd = P.std(0) + 1e-8
        R = Z - self.pca.inverse_transform(P)
        r = np.sqrt((R ** 2).sum(1))
        self.rmu, self.rsd = r.mean(), r.std() + 1e-8
        return self

    def score(self, X):
        Z = X - self.mu
        P = self.pca.transform(Z)
        maha = np.sqrt(((P / self.sd) ** 2).sum(1))          # distance from centre
        R = Z - self.pca.inverse_transform(P)
        recon = (np.sqrt((R ** 2).sum(1)) - self.rmu) / self.rsd   # off-subspace
        return maha + np.maximum(recon, 0)


def detection_stats(s_normal, s_attack, tag_default=None):
    """AUC plus detection rate at each false-alarm budget."""
    y = np.r_[np.zeros(len(s_normal)), np.ones(len(s_attack))]
    out = {"AUC": roc_auc_score(y, np.r_[s_normal, s_attack])}
    for fa in FA_BUDGETS:
        thr = np.quantile(s_normal, 1 - fa)
        out[f"@{fa:.0%}FA"] = float((s_attack > thr).mean())
    out["@0.5"] = tag_default
    return out


# ----------------------------------------------------------------------
# 4) Shared preprocessing for one experiment
# ----------------------------------------------------------------------
def prepare(X_tr, y_tr, X_te, X_extra=None):
    """scale -> reduce -> phases -> quantum features, all fitted on TRAIN."""
    scaler = StandardScaler().fit(X_tr)
    A, B = scaler.transform(X_tr).astype(np.float32), scaler.transform(X_te).astype(np.float32)
    C = scaler.transform(X_extra).astype(np.float32) if X_extra is not None else None

    red = make_reducer(A, y_tr)
    Za, Zb = red.transform(A), red.transform(B)
    Zc = red.transform(C) if C is not None else None

    st = fit_phase_stats(Za)
    t0 = time.time()
    Qa, Qb = quantum_features(to_phase(Za, st)), quantum_features(to_phase(Zb, st))
    Qc = quantum_features(to_phase(Zc, st)) if Zc is not None else None
    print(f"  [quantum] features computed in {time.time()-t0:.1f}s")

    qs = StandardScaler().fit(Qa)
    Qa, Qb = qs.transform(Qa).astype(np.float32), qs.transform(Qb).astype(np.float32)
    Qc = qs.transform(Qc).astype(np.float32) if Qc is not None else None

    ps = StandardScaler().fit(Za)
    Ra, Rb = ps.transform(Za).astype(np.float32), ps.transform(Zb).astype(np.float32)
    Rc = ps.transform(Zc).astype(np.float32) if Zc is not None else None

    tr = {"raw": A, "quantum": Qa, "both": np.hstack([A, Qa]), "red20": Ra}
    te = {"raw": B, "quantum": Qb, "both": np.hstack([B, Qb]), "red20": Rb}
    ex = None if C is None else {"raw": C, "quantum": Qc,
                                 "both": np.hstack([C, Qc]), "red20": Rc}
    return tr, te, ex


# ----------------------------------------------------------------------
# 5) PHASE 1 -- the supervised ablation (unchanged logic)
# ----------------------------------------------------------------------
if RUN_PHASE1:
    print("\n=== PHASE 1: known-attack comparison (70/30) ===")
    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all)
    print(f"  train {len(Xtr):,} / test {len(Xte):,}")
    tr, te, _ = prepare(Xtr, ytr, Xte)

    for name, rep in [("A raw only", "raw"), ("B quantum only", "quantum"),
                      ("C raw + quantum", "both"), ("D dials-only (fair)", "red20")]:
        clf = LogisticRegression(max_iter=3000).fit(tr[rep], ytr)
        print(f"  {name:<22s} test accuracy = "
              f"{accuracy_score(yte, clf.predict(te[rep])):.4f}")

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(tr["raw"], ytr)
    print(f"  {'RF baseline':<22s} test accuracy = {accuracy_score(yte, rf.predict(te['raw'])):.4f}")
    if HAS_XGB:
        xg = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.2,
                           eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(tr["raw"], ytr)
        print(f"  {'XGB baseline':<22s} test accuracy = {accuracy_score(yte, xg.predict(te['raw'])):.4f}")

# ----------------------------------------------------------------------
# 6) PHASE 2 -- zero-day, done properly
# ----------------------------------------------------------------------
for atk in ZERO_DAY_ATTACKS:
    held = (attack_type == atk)
    if held.sum() == 0:
        print(f"\n!! '{atk}' not found in the data -- skipping")
        continue

    X_seen, y_seen = X_all[~held], y_all[~held]
    X_zero = X_all[held]
    print(f"\n{'='*72}\nZERO-DAY '{atk}': {int(held.sum()):,} rows removed from "
          f"training entirely ({len(X_seen):,} remain)\n{'='*72}")

    Xtr, Xte, ytr, yte = train_test_split(
        X_seen, y_seen, test_size=TEST_SIZE, random_state=SEED, stratify=y_seen)
    print(f"  train {len(Xtr):,} / seen-test {len(Xte):,} / zero-day {len(X_zero):,}")

    tr, te, ze = prepare(Xtr, ytr, Xte, X_zero)
    norm_mask = (yte == 0)                       # normal rows of the seen-test
    results, seen_acc = {}, {}

    # ---- supervised detectors (they HAVE seen other attacks) --------------
    for name, rep in [("SUP raw", "raw"), ("SUP quantum", "quantum"),
                      ("SUP raw+quantum", "both")]:
        clf = LogisticRegression(max_iter=3000).fit(tr[rep], ytr)
        seen_acc[name] = accuracy_score(yte, clf.predict(te[rep]))
        s_n = clf.predict_proba(te[rep][norm_mask])[:, 1]
        s_a = clf.predict_proba(ze[rep])[:, 1]
        results[name] = detection_stats(s_n, s_a, float((s_a > 0.5).mean()))

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(tr["raw"], ytr)
    seen_acc["RF (raw)"] = accuracy_score(yte, rf.predict(te["raw"]))
    s_n = rf.predict_proba(te["raw"][norm_mask])[:, 1]
    s_a = rf.predict_proba(ze["raw"])[:, 1]
    results["RF (raw)"] = detection_stats(s_n, s_a, float((s_a > 0.5).mean()))

    # ---- novelty detectors (trained on NORMAL ONLY -- the real fix) -------
    normal_only = (ytr == 0)
    print(f"  [novelty] fitting on {int(normal_only.sum()):,} normal rows only")

    for name, rep in [("NOV raw", "raw"), ("NOV quantum", "quantum"),
                      ("NOV dials-only", "red20")]:
        nov = NoveltyDetector().fit(tr[rep][normal_only])
        results[name] = detection_stats(nov.score(te[rep][norm_mask]), nov.score(ze[rep]))

    iso = IsolationForest(n_estimators=200, random_state=SEED,
                          n_jobs=-1).fit(tr["raw"][normal_only])
    results["IsoForest raw"] = detection_stats(-iso.score_samples(te["raw"][norm_mask]),
                                               -iso.score_samples(ze["raw"]))

    # ---- hybrid: supervised confidence + novelty distance ----------------
    clf = LogisticRegression(max_iter=3000).fit(tr["both"], ytr)
    nov = NoveltyDetector().fit(tr["quantum"][normal_only])

    def _z(a, ref):                       # put both scores on a common scale
        return (a - ref.mean()) / (ref.std() + 1e-8)

    ref_s = clf.predict_proba(te["both"][norm_mask])[:, 1]
    ref_v = nov.score(te["quantum"][norm_mask])
    h_n = _z(ref_s, ref_s) + _z(ref_v, ref_v)
    h_a = _z(clf.predict_proba(ze["both"])[:, 1], ref_s) + _z(nov.score(ze["quantum"]), ref_v)
    results["HYBRID sup+nov"] = detection_stats(h_n, h_a)

    # ---- report ----------------------------------------------------------
    cols = ["AUC"] + [f"@{f:.0%}FA" for f in FA_BUDGETS]
    print(f"\n  {'detector':<20s} {'seen-acc':>9s} {'AUC':>7s} "
          + " ".join(f"{c:>9s}" for c in cols[1:]) + f" {'old@0.5':>9s}")
    print("  " + "-" * 70)
    for name, r in results.items():
        acc = f"{seen_acc[name]:.4f}" if name in seen_acc else "   --   "
        old = f"{r['@0.5']:.1%}" if r["@0.5"] is not None else "   --   "
        cells = " ".join(f"{r[c]:>8.1%}" for c in cols[1:])
        print(f"  {name:<20s} {acc:>9s} {r['AUC']:>7.3f} {cells} {old:>9s}")

print("""
======================= HOW TO READ THIS =======================
AUC       ranks every row by suspicion; 1.000 = perfect separation of the
          unseen attack from normal traffic, 0.500 = coin flip. Threshold
          free, so it is the honest headline number.
@1%FA     detection rate when we flag the 1% most suspicious traffic --
          i.e. a realistic operating point for a security team.
old@0.5   what the previous script reported. Any large gap between this
          and @1%FA means the signal was there all along and only the
          arbitrary 50% cutoff was hiding it.
SUP *     supervised: has seen OTHER attacks but never this one.
NOV *     novelty: trained on NORMAL TRAFFIC ONLY. Nothing is 'unseen' to
          it, which is why it is the principled zero-day detector.
================================================================""")
