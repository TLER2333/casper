import os

import h5py
import numpy as np
from sklearn.metrics import f1_score


def resolve_data_path(dataset_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(dataset_dir, path)


def _decode(values):
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def category_names_from_h5ad(path, column):
    """Read categorical names without loading the expression matrix."""
    with h5py.File(path, "r") as handle:
        node = handle["obs"][column]
        if isinstance(node, h5py.Group):
            return _decode(node["categories"][:])
        return sorted(set(_decode(node[:])))


def evaluate_single_label_logits(logits, targets, valid_candidate_ids, topk=(1, 5, 10, 50, 100)):
    """Compute candidate-masked retrieval and classification metrics."""
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    valid_candidate_ids = np.asarray(valid_candidate_ids, dtype=np.int64)

    if logits.ndim != 2 or len(logits) != len(targets):
        raise ValueError("logits must be [N, C] and targets must contain N labels")
    if valid_candidate_ids.size == 0:
        raise ValueError("the valid non-control candidate set is empty")
    if valid_candidate_ids.max() >= logits.shape[1]:
        raise ValueError("a valid candidate id exceeds the classifier output dimension")

    local_by_global = {int(global_id): i for i, global_id in enumerate(valid_candidate_ids)}
    try:
        local_targets = np.asarray([local_by_global[int(y)] for y in targets])
    except KeyError as exc:
        raise ValueError(f"target id {exc.args[0]} is not in the valid candidate set") from exc

    masked_logits = logits[:, valid_candidate_ids]
    order = np.argsort(-masked_logits, axis=1)
    ranks = np.empty(len(targets), dtype=np.int64)
    for row, target in enumerate(local_targets):
        ranks[row] = int(np.flatnonzero(order[row] == target)[0]) + 1

    pred_global = valid_candidate_ids[order[:, 0]]
    result = {
        "n_examples": int(len(targets)),
        "n_candidates": int(len(valid_candidate_ids)),
        "weighted_f1": float(f1_score(targets, pred_global, average="weighted")),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for k in topk:
        if k <= len(valid_candidate_ids):
            result[f"top{k}_accuracy"] = float(np.mean(ranks <= k))
    for k in (5, 10):
        if k <= len(valid_candidate_ids):
            gain = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
            result[f"ndcg@{k}"] = float(np.mean(gain))
    return result
