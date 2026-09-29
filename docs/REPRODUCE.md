# Reproducing Table 1

This page lists, for every row of Table 1, the data it needs and the command that trains it. The
in-house scenes (Parking lots, Street-view) are not released, so only the Tanks and Temples rows can be
re-run outside Kakao Mobility.

## Contents

- [Datasets](#datasets)
- [Priors](#priors)
- [Training and evaluation](#training-and-evaluation)
- [Row-by-row](#row-by-row)
- [Implementation details](#implementation-details)

## Datasets

| Scene | Paper name | Source |
|:--|:--|:--|
| `Train` | Train | [Tanks and Temples](https://www.tanksandtemples.org/) |
| `Horse` | Horse | [Tanks and Temples](https://www.tanksandtemples.org/) |
| `ladybug6` | Parking lots | in-house, Ladybug6 (front + two side cameras), not released |
| `streetview` | Street-view | in-house, MMS, not released |

Each scene is reconstructed with three SfM pipelines, and all three read the same `images/`:

| SfM directory | Paper name | Pipeline |
|:--|:--|:--|
| `colmap/` | COLMAP | COLMAP (SIFT) |
| `sp-sg/` | SP-SG | SuperPoint + SuperGlue |
| `defree_sfm/` | LoFTR | [Detector-Free SfM](https://github.com/zju3dv/DetectorFreeSfM) (LoFTR matching) |

COLMAP fails on Parking lots and Street-view, so Table 1 has no COLMAP results for them.

Every scene directory follows the COLMAP layout plus one folder of normal priors:

```
<data_root>/<sfm>/<scene>/
├── images/            # undistorted PINHOLE images
├── sparse/0/          # cameras / images / points3D (.bin or .txt)
└── normals/           # <image_name>.npy, float32 (3, H, W), camera-space unit normals
```

Train/test split: every 8th image is a test view (`--eval`, `llffhold=8`).

## Priors

**Normals.** Metric3D v2 ViT-Large (`metric_depth_vit_large_800k.pth`) run on each scene's `images/`.
See [`scripts/metric3d/README.md`](../scripts/metric3d/README.md).

**Depth.** `--depth_use` enables the dense depth prior (`utils/depth_utils.py`). The monocular depth is
scale-aligned to the sparse depth of the projected SfM points.

## Training and evaluation

```bash
python train.py  -s $DATA/<sfm>/<scene> -m output/<sfm>/<scene> --eval [row options]
python render.py -m output/<sfm>/<scene> --skip_train
python metrics.py -m output/<sfm>/<scene>        # writes results.json (PSNR / SSIM / LPIPS)
```

All "Ours" rows share `--lambda_dssim 0.2` and the normal-prior initialization. They differ only in
densification: `--densify_until_iter`, and `--size_threshold_from_iter`, which is the iteration after
which screen-size pruning starts (`-1`: never). `scripts/train_table1.sh` runs all of them.

The 3DGS rows are vanilla [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
with its default settings and `--eval`, on the same scene directories.

## Row-by-row

PSNR / SSIM / LPIPS are the values of Table 1 in the paper.

| Scene | SfM | Options (Ours) | 3DGS | Ours |
|:--|:--|:--|:-:|:-:|
| Train | COLMAP | `--densify_until_iter 10000 --size_threshold_from_iter 8000` | 21.10 / 0.802 / 0.218 | 21.97 / 0.799 / 0.252 |
| Train | SP-SG | defaults | 21.10 / 0.750 / 0.282 | 21.32 / 0.749 / 0.287 |
| Train | LoFTR | `--densify_until_iter 10000 --size_threshold_from_iter 8000` | 20.97 / 0.767 / 0.274 | 21.11 / 0.759 / 0.291 |
| Horse | COLMAP | `--densify_until_iter 20000 --size_threshold_from_iter -1` | 24.18 / 0.889 / 0.239 | 25.50 / 0.903 / 0.153 |
| Horse | SP-SG | `--densify_until_iter 20000 --size_threshold_from_iter -1` | 21.01 / 0.802 / 0.239 | 21.12 / 0.801 / 0.246 |
| Horse | LoFTR | `--densify_until_iter 15000 --size_threshold_from_iter -1` | 23.39 / 0.870 / 0.174 | 24.80 / 0.881 / 0.165 |
| Parking lots | SP-SG | `--densify_until_iter 10000 --size_threshold_from_iter 10000` | 28.08 / 0.828 / 0.424 | 28.37 / 0.829 / 0.427 |
| Parking lots | LoFTR | defaults | 29.33 / 0.842 / 0.410 | 29.07 / 0.840 / 0.414 |
| Street-view | SP-SG | `--densify_until_iter 3000 --size_threshold_from_iter 1000` | 18.33 / 0.468 / 0.410 | 19.00 / 0.479 / 0.492 |
| Street-view | LoFTR | defaults | 22.28 / 0.729 / 0.331 | 22.49 / 0.722 / 0.340 |

Mean reprojection error (MRE) of each SfM reconstruction, from Table 1:

| Scene | COLMAP | SP-SG | LoFTR |
|:--|:-:|:-:|:-:|
| Train | 0.75 | 1.40 | 0.79 |
| Horse | 0.71 | 1.31 | 0.80 |
| Parking lots | invalid | 1.37 | 0.64 |
| Street-view | invalid | 1.09 | 0.56 |

3DGS optimization is not deterministic on GPU, so a re-run lands near these values rather than on them.

## Implementation details

Values as set in this code:

| Setting | Value |
|:--|:--|
| `--lambda_dssim` (λ₁) | 0.2 |
| `--lambda_dnormal` (weight of L_normal) | 1e-4 |
| `--lambda_normal_axis` (λ in L_normal = λ L_axis + (1−λ) L_scale) | 0.8 |
| `--lambda_depth` (weight of L_depth, with `--depth_use`) | 0.5 |
| iterations | 30,000 |
| normal / depth network | Metric3D v2 ViT-Large |
