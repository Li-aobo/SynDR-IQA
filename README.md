# SynDR-IQA
This is the official PyTorch implementation for the NeurIPS 2025 paper **"Towards Syn-to-Real IQA: A Novel Perspective on Reshaping Synthetic Data Distributions"**.


## Requirement
+ Python 3.6
+ pytorch 1.4.0
+ torchvision 0.5.0

## Usage

### 1. Prepare datasets

Download **`ref_imgs`** (reference images of KADID-10k) and **`kadid_add81`**, and place them under the `KADID_10k` directory.

- Google Drive: https://drive.google.com/file/d/1lnkxkDG5z6KG4u5Yj-T7VX7ivMCwxLpk/view?usp=sharing  
- 百度网盘：链接 https://pan.baidu.com/s/12WqbuznpJBmXa5nGIsPilQ?pwd=hizm 提取码 hizm

You should obtain the following folder structure (here `root` is the path to your dataset root directory):

```text  
root/  
├── KADID_10k  
│   ├── images  
│   ├── ref_imgs          # NEW!!!  
│   └── kadid_add81       # NEW!!!  
│       ├── ref_imgs  
│       └── dist_imgs  
├── ChallengeDB_release   # LIVE Challenge (LIVEC)  
├── KonIQ_10k  
└── BID_512  
```


### 2. **Extract reference features (build RefSet for DDCUp)**

Run the following script to compute reference features, or skip this step if you use the provided RefSet:

```bash
python get_ref_feature.py --root /path/to/root
```

### 3. **Train and test**

Use `train_test_IQA_cross_full.py` to run cross-database experiments.  
For example, to use BID as the test database:

```bash
python train_test_IQA_cross_full.py --test_db bid --root /path/to/root
```

Replace `bid` with other dataset names as needed (e.g., `koniq-10k`, `livec`).

### 4. **Pre-trained models**

We provide the pre-trained models corresponding to the experimental results reported in the paper:

- Google Drive: https://drive.google.com/drive/folders/1vOwhC-eqkoYO_0mrlHYRPaHWckyMKlMa?usp=sharing
- 百度网盘：链接: https://pan.baidu.com/s/1FQhtunqE8kpitJzz1zXydw?pwd=7ipx 提取码: 7ipx
   

## Citation
If you find our code helpful for your research, please consider citing:

```
@inproceedings{li2025towards,
  title={Towards Syn-to-Real IQA: A Novel Perspective on Reshaping Synthetic Data Distributions},
  author={Li, Aobo and Wu, Jinjian and Liu, Yongxu and Li, Leida and Dong, Weisheng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
