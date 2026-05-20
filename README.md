# ViMeMo

ViMeMo is a small Python package and notebook workflow for exploratory viral membrane morphometry from cryo-EM maps.

The current codebase covers the workflow extracted from `test_memcurv.ipynb`:

- load `.mrc` maps
- survey density along radial directions
- detect and clean maxima
- assign membrane leaflets
- diagnose and remove outliers
- calculate spacing and curvature morphometrics
- generate diagnostic 3D plots and Hammer-projection plots

## Repository layout

- `vimemo/`: reusable package code
- `vimemo_run.ipynb`: thin runner notebook for the current workflow
- `test_memcurv.ipynb`: original exploratory notebook the package was extracted from
- `my_output_notebook.ipynb`: legacy downstream analysis notebook kept for reference
- `hammer_contour_griddata.ipynb`: legacy Hammer-plot exploration notebook kept for reference

## Installation

Create or activate your environment, then install the package in editable mode:

```bash
pip install -e .
```

Optional overlay support in Hammer plots can use `alphashape` and `shapely` if installed. The plotting code degrades gracefully without them.

## Basic usage

In Python or Jupyter:

```python
from vimemo import (
    CryoEMSurvey,
    MaximaModel,
    MembraneMorphometry,
    ModelMask,
    load_mrc_map,
    plot_membrane_hammer,
    plot_model_mask_hammer,
)
```

The main interactive entry point for now is [vimemo_run.ipynb](./vimemo_run.ipynb).

## Notes

- Raw `.mrc` maps are ignored by git in this repository.
- The package is still exploratory and notebook-driven; the runner notebook is the supported interface for now.
