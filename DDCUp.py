import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

from base_tools import load_pkl, dump_pkl, normalize_labels


def _softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _process_matrix(A, x):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    original_indices = np.arange(A.shape[0])

    while True:
        min_value = np.min(A)
        if min_value >= x:
            break

        min_indices = np.unravel_index(np.argmin(A), A.shape)

        A = np.delete(A, min_indices[0], axis=0)
        A = np.delete(A, min_indices[0], axis=1)
        original_indices = np.delete(original_indices, min_indices[0])

        if A.size == 0 or A.shape[0] <= 1:
            break

    return original_indices, A


def DDCUp(
    res_path: str,
    kadid_root: str,
    sel_types: np.ndarray,
    patch_num: int,
    cache_path: str = None,
):

    add_root = os.path.join(kadid_root, "kadid_add81")
    dmos_csv = os.path.join(kadid_root, "dmos.csv")

    Tres = load_pkl(os.path.join(res_path, "train_ref.pkl"))      # (Tnames, Tfeatures)
    Ares = load_pkl(os.path.join(res_path, "additional_ref.pkl")) # (Anames, Afeatures)
    Tnames = Tres[0]
    Anames = Ares[0]

    Tfeat = np.concatenate(Tres[1], axis=0)
    Afeat = np.concatenate(Ares[1], axis=0)

    AFds = cdist(Afeat, Tfeat, metric="cosine")

    TFds = squareform(pdist(Tfeat, metric="cosine"))
    maxT = np.max(TFds, axis=1)
    maxA = np.max(AFds, axis=1)
    minA = np.min(AFds, axis=1)
    np.fill_diagonal(TFds, 1)
    minT = np.min(TFds, axis=1)

    aaa = np.logical_and(minA > np.median(minT), maxA < np.max(maxT))

    A1Fds = squareform(pdist(Afeat[aaa, :], metric="cosine"))
    np.fill_diagonal(A1Fds, 1)
    fin_sel_idxs, _ = _process_matrix(A1Fds, np.median(minT))

    Anames = np.array(Anames)[aaa][fin_sel_idxs]
    AFds = AFds[aaa, :][fin_sel_idxs, :]

    # print(len(Anames), end=" ")

    data = pd.read_csv(dmos_csv)
    data["dmos"] = normalize_labels(data["dmos"].values)

    sim_idxs = np.argsort(AFds, axis=1)
    n_top = 5
    TopNsim_idxs = sim_idxs[:, :n_top]

    Adist_names = []
    Admos = []

    for i in range(len(Anames)):
        simFs = AFds[i, TopNsim_idxs[i]]
        sel_idxs = TopNsim_idxs[i][simFs - simFs[0] < 0.05]
        simFs = simFs[simFs - simFs[0] < 0.05]

        if isinstance(simFs, np.ndarray):
            weights = _softmax(simFs)
            dmos_list = []
            for idx in sel_idxs:
                name = Tnames[idx]
                sel_rows = data[data["ref_img"] == name]
                dmos_list.append(sel_rows["dmos"].values)
            dmos = np.sum(np.array(dmos_list) * weights.reshape([-1, 1]), axis=0)
        else:
            name = Tnames[sel_idxs]
            sel_rows = data[data["ref_img"] == name]
            dmos = sel_rows["dmos"].values

        base_name = Anames[i].split(".")[0]
        Adist_names += [
            f"{base_name}_{ti:02d}_{lv:02d}.bmp"
            for ti in range(1, 26)
            for lv in range(1, 6)
        ]
        Admos.append(dmos)

    Adist_names = np.array(Adist_names)
    Admos1 = np.array(Admos).astype(np.float32).flatten()

    dist_lvs = np.array([1, 3, 5]).reshape([1, -1])
    sel_imgs = np.arange(len(Anames)).reshape([-1, 1])
    sel_dists = dist_lvs + (sel_types - 1) * 5 - 1
    sel_idx = (sel_dists.reshape(1, -1) + sel_imgs * 125).flatten().tolist()

    imgnames = Adist_names[sel_idx].tolist()
    labels = Admos1[sel_idx]

    samples = []
    for i, imgname in enumerate(imgnames):
        full_path = os.path.join(add_root, "dist_imgs", imgname)
        for _ in range(patch_num):
            samples.append((imgname, full_path, labels[i]))

    if cache_path is not None:
        dump_pkl(cache_path, samples)

    return samples


def upsample(
    res_path: str,
    kadid_root: str,
    sel_types: np.ndarray,
    patch_num: int,
    cache_path: str,
):

    if os.path.isfile(cache_path):
        return load_pkl(cache_path)

    samples = DDCUp(
        res_path=res_path,
        kadid_root=kadid_root,
        sel_types=sel_types,
        patch_num=patch_num,
        cache_path=cache_path,
    )
    return samples