# ViMeMo

ViMeMo (**Viral Membrane Morphometry**) is a small Python package and notebook workflow for exploratory viral membrane morphometry from cryo-EM maps.

The current codebase covers a notebook-driven workflow for:

- load `.mrc` maps
- survey density along radial directions
- detect and clean maxima
- assign membrane leaflets
- diagnose and remove outliers
- calculate spacing and curvature morphometrics
- generate diagnostic 3D plots for different stages of cleanup
- generate final Hammer-projection plots

The package is intentionally notebook-driven at this stage. The main supported entry point is [vimemo_run.ipynb](./vimemo_run.ipynb), where parameters remain visible and editable at each step.

## Repository layout

- `vimemo/`: reusable package code
- `vimemo_run.ipynb`: thin runner notebook for the current workflow
- `my_output_notebook.ipynb`: legacy downstream analysis notebook kept for reference
- `hammer_contour_griddata.ipynb`: legacy Hammer-plot exploration notebook kept for reference

## Installation

Create or activate your environment, then install the package in editable mode:

```bash
pip install -e .
```

Optional overlay support in Hammer plots can use `alphashape` and `shapely` if installed. The plotting code degrades gracefully without them.

## Main objects

### `load_mrc_map`

`load_mrc_map(...)` is the low-level file loader. It reads an `.mrc` map from disk and returns the 3D NumPy array used by the rest of the workflow.

### `ModelMask`

`ModelMask` converts a loaded mask-like map into a voxel dataframe with:

- Cartesian coordinates
- spherical coordinates in the same convention used by `CryoEMSurvey`
- stored Cartesian and spherical coordinate limits

It is mainly used to inspect angular coverage and to provide visual Hammer-plot overlays such as ASU or symmetry-region outlines.

### `CryoEMSurvey`

`CryoEMSurvey` is the first core analysis object. It takes the cryo-EM density map and:

- stores survey radii and angular limits
- generates radial direction vectors on a spherical grid
- samples density values along each radial direction

This stage transforms a raw 3D map into a structured set of 1D radial density profiles.

### `MaximaModel`

`MaximaModel` detects peaks in the surveyed radial profiles and turns them into 3D candidate membrane points. It is responsible for:

- peak calling along each direction
- initial clustering
- voxel-connectivity filtering
- per-direction cleanup
- restoring the original detected maxima with `recover_initial_maxima_model()`

Conceptually, this is the stage where the workflow goes from surveyed density profiles to a cleaned point cloud likely to represent the membrane.

### `MembraneMorphometry`

`MembraneMorphometry` starts from a cleaned `MaximaModel` point cloud and adds membrane-specific interpretation. It is responsible for:

- assigning leaflet identity
- diagnosing outliers
- removing outlier points
- calculating inter-leaflet spacing
- calculating local curvature morphometrics

This is the stage where the point cloud becomes a membrane surface description with biological meaning.

### Hammer plotting functions

The package currently exposes two Hammer-projection plotting helpers:

- `plot_model_mask_hammer(...)`
- `plot_membrane_hammer(...)`

`plot_model_mask_hammer(...)` is for footprint-style mask plots.

`plot_membrane_hammer(...)` is for membrane-derived scalar fields such as:

- spacing
- mean curvature
- Gaussian curvature

For membrane plots, a leaflet must be specified so that only one surface is plotted at a time. A `ModelMask` can be added only as a visual overlay.

## Notebook workflow

The runner notebook follows a deliberately explicit step-by-step flow.

### 1. Load maps

The notebook first loads:

- the main cryo-EM density map
- any mask-like maps used later for angular inspection or Hammer overlays

### 2. Build `ModelMask` objects

Mask maps are converted into `ModelMask` objects so you can:

- inspect their voxel dataframe
- inspect Cartesian limits
- inspect spherical limits
- reuse them later as Hammer-plot overlays

### 3. Build a `CryoEMSurvey`

The main density map is wrapped in `CryoEMSurvey`, where you define:

- pixel size
- radial survey range
- angular limits
- angular sampling step

The notebook then computes the direction vectors and samples density along them.

### 4. Build and clean a `MaximaModel`

The surveyed density data is converted into a `MaximaModel`, then the notebook runs an explicit exploratory cleaning sequence:

- initial clustering
- cluster selection
- voxel connectivity grouping
- connected-component selection
- per-direction cleanup

This stage is still meant to be interactive. The notebook exposes the parameters directly so they can be inspected and adjusted.

### 5. Build `MembraneMorphometry`

The cleaned maxima dataframe is passed into `MembraneMorphometry`, which immediately assigns leaflet labels. At this point the notebook can already show leaflet-colored diagnostic plots.

### 6. Diagnose and remove outliers at the leaflet level

Outliers are diagnosed within each leaflet using local neighborhood distances. The notebook lets you inspect the resulting outlier score before removing flagged points.

### 7. Calculate morphometrics

After cleanup, the notebook calculates:

- spacing between paired leaflet points
- principal curvatures
- derived mean curvature
- derived Gaussian curvature

This ordering is intentional: morphometrics are meant to be calculated after the main outlier cleanup.

### 8. Generate Hammer plots

Finally, the notebook can project membrane scalar fields onto a Hammer map and optionally draw a `ModelMask` outline on top for reference.

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

## Notes

- Raw `.mrc` maps are ignored by git in this repository.
- The package is still exploratory and notebook-driven; the runner notebook is the supported interface for now.
