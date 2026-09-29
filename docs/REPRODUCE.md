# Reproducing Table 1

This page maps every row of Table 1 to the data it needs, the command that trains it, and the value its
trained model gives. The in-house scenes (Parking lots, Street-view) are not released, so only the
Tanks and Temples rows can be re-run outside Kakao Mobility.

## Contents

- [Datasets](#datasets)
- [Priors](#priors)
- [Training and evaluation](#training-and-evaluation)
- [Row-by-row](#row-by-row)
- [What the paper text does not state](#what-the-paper-text-does-not-state)

## Datasets

| Scene | Paper name | Source |
|:--|:--|:--|
| `Train` | Train | [Tanks and Temples](https://www.tanksandtemples.org/) (training set) |
| `Horse` | Horse | [Tanks and Temples](https://www.tanksandtemples.org/) (intermediate set) |
| `ladybug6` | Parking lots | in-house, Ladybug6 (front + two side cameras), not released |
| `streetview` | Street-view | in-house, MMS, not released |

Each scene is reconstructed three times, once per SfM pipeline, and every run of that scene reads the
same `images/`:

| SfM directory | Paper name | Pipeline |
|:--|:--|:--|
| `colmap/` | COLMAP | COLMAP (SIFT) |
| `sp-sg/` | SP-SG | SuperPoint + SuperGlue |
| `defree_sfm/` | LoFTR | [Detector-Free SfM](https://github.com/zju3dv/DetectorFreeSfM) (LoFTR matching) |

COLMAP fails on `ladybug6` and `streetview`, so those two cells of Table 1 are empty.

Every scene directory follows the COLMAP layout plus one folder of normal priors:

```
<data_root>/<sfm>/<scene>/
├── images/            # undistorted PINHOLE images
├── sparse/0/          # cameras / images / points3D (.bin or .txt)
└── normals/           # <image_name>.npy, float32 (3, H, W), camera-space unit normals
```

Train/test split: every 8th image is a test view (`--eval`, `llffhold=8`).

## Priors

**Normals.** The normal priors come from Metric3D v2 ViT-Large (`metric_depth_vit_large_800k.pth`),
run on each scene's `images/`. See [`scripts/metric3d/README.md`](../scripts/metric3d/README.md) for the
command and the few lines that write the `.npy` files.

**Depth.** `--depth_use` turns on the dense depth prior (`utils/depth_utils.py`). The monocular depth
is scale-aligned to the sparse depth of the projected SfM points. **The depth prior is off in every
Table 1 run**: the `cfg_args` of all eleven trained models record `depth_use=False`.

## Training and evaluation

```bash
python train.py  -s $DATA/<sfm>/<scene> -m output/<sfm>/<scene> --eval [row options]
python render.py -m output/<sfm>/<scene> --skip_train
python metrics.py -m output/<sfm>/<scene>        # writes results.json (PSNR / SSIM / LPIPS-VGG)
```

The options shared by every "Ours" row are the defaults: `--lambda_dssim 0.2 --lambda_dnormal 1e-4
--lambda_normal_axis 0.8`, normal-prior initialization on, and depth prior off. The rows differ only in
densification: `--densify_until_iter` and `--size_threshold_from_iter`. The latter is the iteration
after which screen-size pruning (`max_screen_size=20`) starts, and `-1` disables it.

The 3DGS rows are vanilla [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
trained with its default settings and `--eval` on the same scene directories.

The research runs were repeated 4–8 times per configuration with the same seed. The model kept for
Table 1 is one of those repeats, because CUDA rasterization is not deterministic.

## Row-by-row

"Model" is the value that the kept model's `results.json` gives, rounded the way Table 1 is.
✅ = equal to Table 1. ⚠️ = see the note.

### Ours

| Scene | SfM | Options | Table 1 PSNR / SSIM / LPIPS | Model | |
|:--|:--|:--|:-:|:-:|:-:|
| Train | COLMAP | `--densify_until_iter 10000 --size_threshold_from_iter 8000` | 21.97 / 0.799 / 0.252 | 21.97 / 0.799 / 0.252 | ✅ |
| Train | SP-SG | see note 1 | 21.32 / 0.749 / 0.287 | 21.32 / 0.749 / 0.287 | ✅ ¹ |
| Train | LoFTR | `--densify_until_iter 10000 --size_threshold_from_iter 8000` | 21.11 / 0.759 / 0.291 | 21.11 / 0.759 / 0.292 | ⚠️ ³ |
| Horse | COLMAP | `--densify_until_iter 20000 --size_threshold_from_iter -1` | 25.50 / 0.903 / 0.153 | 25.50 / 0.903 / 0.153 | ✅ ² |
| Horse | SP-SG | `--densify_until_iter 20000 --size_threshold_from_iter -1` | 21.12 / 0.801 / 0.246 | 21.12 / 0.801 / 0.246 | ✅ |
| Horse | LoFTR | `--densify_until_iter 15000 --size_threshold_from_iter -1` | 24.80 / 0.881 / 0.165 | 24.80 / 0.882 / 0.165 | ⚠️ ³ |
| Parking lots | SP-SG | `--densify_until_iter 10000 --size_threshold_from_iter 10000` | 28.37 / 0.829 / 0.427 | 28.36 / 0.830 / 0.427 | ⚠️ ³ |
| Parking lots | LoFTR | see note 1 | 29.07 / 0.840 / 0.414 | 29.08 / 0.841 / 0.412 | ⚠️ ¹ ³ |
| Street-view | SP-SG | `--densify_until_iter 3000 --size_threshold_from_iter 1000` | 19.00 / 0.479 / 0.492 | 19.00 / 0.479 / 0.492 | ✅ |
| Street-view | LoFTR | see note 1 | 22.49 / 0.722 / 0.340 | 22.49 / 0.722 / 0.340 | ✅ ¹ |

### 3DGS

| Scene | SfM | Table 1 PSNR / SSIM / LPIPS | Model | |
|:--|:--|:-:|:-:|:-:|
| Train | COLMAP | 21.10 / 0.802 / 0.218 | 22.01 / 0.809 / 0.232 | ⚠️ ⁴ |
| Train | SP-SG | 21.10 / 0.750 / 0.282 | 21.10 / 0.751 / 0.282 | ⚠️ ³ |
| Train | LoFTR | 20.97 / 0.767 / 0.274 | 20.97 / 0.767 / 0.274 | ✅ |
| Horse | COLMAP | 24.18 / 0.889 / 0.239 | 24.45 / 0.896 / 0.157 | ⚠️ ⁴ |
| Horse | SP-SG | 21.01 / 0.802 / 0.239 | 21.01 / 0.802 / 0.239 | ✅ |
| Horse | LoFTR | 23.39 / 0.870 / 0.174 | 23.39 / 0.870 / 0.174 | ✅ |
| Parking lots | SP-SG | 28.08 / 0.828 / 0.424 | 28.08 / 0.828 / 0.424 | ✅ |
| Parking lots | LoFTR | 29.33 / 0.842 / 0.410 | 29.33 / 0.842 / 0.407 | ⚠️ ³ |
| Street-view | SP-SG | 18.33 / 0.468 / 0.410 | 18.33 / 0.468 / 0.410 | ✅ |
| Street-view | LoFTR | 22.28 / 0.729 / 0.331 | 22.28 / 0.729 / 0.331 | ✅ |

**Notes**

1. These three models were trained with the research `train.py`, which was edited again after the
   paper. The densification settings it had at training time are therefore not recorded. Its later
   state (`--densify_until_iter 15000 --size_threshold_from_iter -1`, no normal-prior initialization) is
   not necessarily what produced these numbers.
2. The training command for this model is not in the run log. The settings are those of the script
   that the other Horse runs of the same batch used.
3. The model and Table 1 differ in the last digit (at most 0.01 PSNR, 0.001 SSIM, 0.003 LPIPS).
4. The COLMAP 3DGS models kept for Train and Horse have no `results.json`. The values shown were computed
   from their test renderings with this repository's `metrics.py` code, and the same code reproduces the
   SP-SG Horse 3DGS `results.json` exactly. Table 1's values for these two cells therefore came from other
   runs of the same configuration, which were not kept with their renderings.

## What the paper text does not state

These are properties of the code that produced Table 1, recorded so that a re-run matches it:

- The normal-consistency loss weight is `1e-4`, and the axis/scale split `λ` is `0.8`.
- The depth prior (`L_depth`) is off in every Table 1 run.
- Densification settings are chosen per scene (table above).
- The normal-prior initialization (`utils/norminit_utils.py`) is released as it was found in the research
  code. In that state, the per-view loop that samples a normal for each Gaussian and sets its rotation is
  disabled (lines 241–316 are inside a string literal), so the function only flattens every Gaussian: the
  first scale axis is set to 1e-5, and the other two to 0.1 where the first was non-negative. The file was
  edited while the Table 1 models were being trained, so which state each model saw is not recorded. The
  full loop is the one in [VEGS](https://github.com/deepshwang/vegs).
