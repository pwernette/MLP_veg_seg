# ML Vegetation Filter – QGIS Plugin

A QGIS Processing plugin that wraps the
[ML Vegetation Filter](https://github.com/pwernette/point_cloud_vegetation_filtering)
toolset by Phillipe A. Wernette
(doi: [10.5281/zenodo.10966854](https://doi.org/10.5281/zenodo.10966854)).

It classifies LAS/LAZ point cloud points as **vegetation** or **bare-Earth**
(or any two visually-distinguishable classes) using RGB-derived vegetation
indices and a TensorFlow MLP neural network.

---

## Bugs fixed from the original scripts

| File | Bug | Fix |
|------|-----|-----|
| `ML_vegfilter.py` line 259 | Missing comma in `predict_reclass_write()` call after `geom_rad=…` → `SyntaxError` | Comma added |
| `src/fileio.py` line 554 | `globals()[('incloud.prob0')]` sets a key in the global dict but never assigns to the `incloud` object → probability extra-bytes silently not written | Replaced with `setattr(incloud, 'prob'+str(i), outdat_pred[:,i])` |

---

## Three algorithms

| # | Algorithm | Equivalent script |
|---|-----------|-------------------|
| 1 | **Train vegetation filter model** | `ML_veg_train.py` |
| 2 | **Reclassify point cloud with saved model** | `ML_veg_reclass.py` |
| 3 | **Train model and reclassify (combined)** | `ML_vegfilter.py` |

All three appear under **Processing Toolbox → ML Vegetation Filter**.

---

## Installation

### 1 – Install Python dependencies

The plugin requires packages not bundled with QGIS. Install them into the
Python environment that QGIS uses.

**Linux (OSGeo4W or conda-based QGIS) – GPU support:**
```bash
conda env create -f ML_veg_filter_env_linux.yml
conda activate mlvegfilter_qgis
```

**Windows (OSGeo4W shell) – CPU only:**
```bat
pip install tensorflow laspy lazrs pandas scikit-learn tqdm matplotlib pydot
```

**Windows – GPU support (requires matching CUDA 11.2 + cuDNN 8.1):**
```bash
conda env create -f ML_veg_filter_env.yml
conda activate mlvegfilter_qgis
```

> **Tip:** Make sure you use `pip install` or `conda install` into the
> **same** Python that QGIS is using. You can check by running
> `import sys; print(sys.executable)` from the QGIS Python console.

### 2 – Copy the plugin folder

Copy the `ml_veg_filter_qgis/` folder into your QGIS plugins directory:

| OS | Path |
|----|------|
| Linux | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |
| Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` |
| macOS | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |

### 3 – Enable the plugin

1. Open QGIS → **Plugins → Manage and Install Plugins**
2. Search for **ML Vegetation Filter** and tick the checkbox
3. The algorithms appear under **Processing Toolbox → ML Vegetation Filter**

---

## Quick-start workflow

### Option A – Two separate steps

**Step 1 – Prepare two (or more) labelled LAS/LAZ files:**
- File 1 → class 0 (e.g. bare Earth)
- File 2 → class 1 (e.g. vegetation)

**Step 2 – Run Algorithm 1 (Train).** Paste both file paths (one per line),
choose a vegetation index preset and hyperparameters, and pick an output folder.

**Step 3 – Run Algorithm 2 (Reclassify).** Point to the `.keras`/`.h5` model
file and the target point cloud to classify.

### Option B – One step

Run **Algorithm 3 (Train + Reclassify)**. Provide the training files, the
target cloud, and an output folder. Model training and reclassification happen
in sequence without reloading the model from disk.

---

## Vegetation index presets

| Preset | Features used |
|--------|--------------|
| `rgb` | Normalised R, G, B only |
| `simple` | R, G, B + ExR, ExG, ExB, ExGR |
| `all` | All 10 RGB-derived indices |
| `custom` | User-specified comma-separated list |

Individual index tokens for `custom`:
`exr`, `exg`, `exb`, `exgr`, `ngrdi`, `mgrvi`, `gli`, `rgbvi`, `ikaw`, `gla`,
`xyz` (normalised XYZ coordinates),
`sd` (per-axis 3-D standard deviation),
`3d` (combined 3-D standard deviation).

> **Note:** `sd` and `3d` are very compute-intensive (point-wise KD-tree search).
> Only use them if you have evidence they improve accuracy for your dataset.

---

## Outputs

### Training (Algorithms 1 and 3)

| File | Description |
|------|-------------|
| `<name>_FULL_MODEL.keras` / `.h5` | Complete trained model |
| `<name>_BEST.keras` / `.h5` | Best checkpoint (lowest val_loss) |
| `<name>_MODEL_WEIGHTS.h5` | Weights only |
| `<name>_MODEL_SUMMARY.txt` | Architecture, inputs, evaluation metrics |
| `<name>_LOG.csv` | Per-epoch training log |
| `<name>_PLOT_TRAINING.png` | Accuracy and loss curves (optional) |

### Reclassification (Algorithms 2 and 3)

| File | Description |
|------|-------------|
| `results_<date>/<input>_<model>.laz` | Reclassified point cloud |

The output LAZ file has an updated `classification` field and, if enabled,
extra-bytes fields `prob0`, `prob1`, … containing per-class probabilities.

---

## Citation

```bibtex
@software{Wernette2024,
  author  = {Wernette, Phillipe A.},
  title   = {Segmenting Vegetation from bare-Earth in High-relief and Dense
             Point Clouds using Machine Learning},
  year    = {2024},
  doi     = {10.5281/zenodo.10966854},
  url     = {https://doi.org/10.5281/zenodo.10966854},
  version = {1.00},
}
```
