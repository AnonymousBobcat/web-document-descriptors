"""
Minimal timing for the paper's claim that LR training/inference is
'computationally minimal compared to an encoder classifier'.

Reports, for the configuration actually used in the paper
(harmonized descriptors, min_freq=1, i.e. full descriptor vocabulary):
  - hardware (CPU model + core count)
  - feature dimensionality (vocab size) per task
  - fit + inference wall-clock per task and total

Run this on YOUR machine against YOUR real 80/20 split. The number
printed here is the one to put in the paper; do NOT use a number
produced on any other machine.

Notes:
  - n_jobs is FIXED (not -1) so the figure is reproducible and the
    reported core count is meaningful. Single-core (N_JOBS=1) gives the
    most conservative 'minimal compute' claim. Set to a higher value if
    you prefer to report a realistic multi-core wall-clock; just report
    that core count.
  - Wall-clock, not CPU time. Fine here because the LR-vs-encoder gap is
    orders of magnitude, not a few percent.
"""

import json
import platform
import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------- config: the paper's chosen setup ----------
TRAIN_PATH = "data/80_topic_format_edu.jsonl"
TEST_PATH = "data/20_topic_format_edu.jsonl"
DESCRIPTOR_KEY = "harmonized_descriptors"  # the reported setting
MIN_FREQ = 1  # full vocabulary, no filtering
TASKS = ["label", "topic", "format"]
SEED = 42
N_JOBS = 1  # fixed for reproducibility; report this core count


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_descriptors(doc):
    return set(doc[DESCRIPTOR_KEY])


def build_descriptor_features(train, test, min_freq):
    freq = Counter()
    for d in train:
        for x in get_descriptors(d):
            freq[x] += 1
    vocab = sorted([w for w, n in freq.items() if n >= min_freq])
    idx = {w: i for i, w in enumerate(vocab)}

    def to_matrix(docs):
        rows, cols = [], []
        for r, d in enumerate(docs):
            for x in get_descriptors(d):
                if x in idx:
                    rows.append(r)
                    cols.append(idx[x])
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(len(docs), len(vocab)))

    return to_matrix(train), to_matrix(test), vocab


# ---------- hardware ----------
try:
    import subprocess

    cpu = subprocess.check_output(["lscpu"]).decode()
    cpu_model = next(
        (l.split(":", 1)[1].strip() for l in cpu.splitlines() if "Model name" in l),
        platform.processor() or "unknown",
    )
except Exception:
    cpu_model = platform.processor() or "unknown"

print(f"CPU: {cpu_model}")
print(f"n_jobs (cores used): {N_JOBS}")
print(f"descriptor_key: {DESCRIPTOR_KEY}, min_freq: {MIN_FREQ}\n")

train = load_jsonl(TRAIN_PATH)
test = load_jsonl(TEST_PATH)
print(f"train={len(train)} test={len(test)}\n")

total_fit = 0.0
total_pred = 0.0

for task in TASKS:
    le = LabelEncoder()
    y_tr = le.fit_transform([d[task] for d in train])
    y_te = le.transform([d[task] for d in test])
    n_classes = len(le.classes_)

    X_tr, X_te, vocab = build_descriptor_features(train, test, MIN_FREQ)

    clf = LogisticRegression(
        solver="saga", max_iter=1000, random_state=SEED, n_jobs=N_JOBS
    )

    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = clf.predict(X_te)
    t_pred = time.perf_counter() - t0

    total_fit += t_fit
    total_pred += t_pred
    print(
        f"[{task}] classes={n_classes} vocab={len(vocab)} "
        f"fit={t_fit:.2f}s infer={t_pred:.3f}s"
    )

print(
    f"\nTOTAL across {len(TASKS)} tasks: "
    f"fit={total_fit:.2f}s infer={total_pred:.3f}s "
    f"(fit+infer={total_fit + total_pred:.2f}s)"
)
print("\n-> Report: CPU model, core count, vocab sizes, and the TOTAL line.")
