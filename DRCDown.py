# sample_selector.py
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform


def _union_find(groups_idx: np.ndarray):
    parent = {}
    rank = {}

    def make_set(x):
        parent[x] = x
        rank[x] = 0

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_x] = root_y
            if rank[root_x] == rank[root_y]:
                rank[root_y] += 1

    element_to_id = {}
    next_id = 0

    for col in groups_idx.T:
        for element in col:
            if element not in element_to_id:
                element_to_id[element] = next_id
                make_set(next_id)
                next_id += 1

    for col in groups_idx.T:
        first_element_id = element_to_id[col[0]]
        for element in col[1:]:
            union(first_element_id, element_to_id[element])

    final_groups = {}
    for element, elem_id in element_to_id.items():
        root_id = find(elem_id)
        final_groups.setdefault(root_id, set()).add(element)

    return [list(v) for v in final_groups.values()]


def _filter_samples(fea_ds: np.ndarray, gt_ds: np.ndarray, sel_num: int = 20):
    n = gt_ds.shape[0]

    rows, cols = np.meshgrid(np.arange(n), np.arange(n))
    rows_flat = rows.ravel()
    cols_flat = cols.ravel()
    idx_pairs = np.vstack([rows_flat, cols_flat])  # shape: (2, n*n)

    mask = np.logical_and(gt_ds < 1, fea_ds > 0.9)
    np.fill_diagonal(mask, False)
    mask_flat = mask.ravel()

    selected_idx = idx_pairs[:, mask_flat]  # shape: (2, M)

    groups = _union_find(selected_idx)
    group_lens = np.array([len(g) for g in groups])

    if groups:
        all_group_elems = set().union(*groups)
        filtered_idxs = set(range(n)) - all_group_elems
    else:
        filtered_idxs = set(range(n))

    for group, length in zip(groups, group_lens):
        if length > sel_num:
            keep = random.sample(group, max(sel_num, int(length / 2)))
        else:
            keep = group
        filtered_idxs.update(keep)

    filtered_idxs = sorted(filtered_idxs)
    # print(len(filtered_idxs), end=" ")

    return np.array(filtered_idxs, dtype=int)


def DRCDown(gts, fs):
    gts = np.array(gts)
    fs = np.concatenate(fs, axis=0)  # (N, feat_dim)

    fea_ds = 1 - squareform(pdist(fs, metric="cosine"))
    gt_ds = np.abs(gts[:, None] - gts[None, :])

    filtered_idxs = _filter_samples(fea_ds, gt_ds)
    return filtered_idxs