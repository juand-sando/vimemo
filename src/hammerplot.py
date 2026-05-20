from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata

from .membranemorphometry import MembraneMorphometry
from .modelmask import ModelMask


def hammer_projection(
    phi: np.ndarray,
    theta: np.ndarray,
    phi_center: float = np.pi,
    theta_center: float = np.pi / 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to Hammer projection coordinates.
    """
    phi_centered = phi - phi_center
    theta_shifted = theta - theta_center + np.pi / 2
    latitude = np.pi / 2 - theta_shifted

    denominator = np.sqrt(
        1 + np.cos(latitude) * np.cos(phi_centered / 2)
    )

    x_coord = (
        2
        * np.sqrt(2)
        * np.cos(latitude)
        * np.sin(phi_centered / 2)
        / denominator
    )
    y_coord = np.sqrt(2) * np.sin(latitude) / denominator

    return x_coord, y_coord


def create_segment_boundary(
    phi_lim: tuple[float, float],
    theta_lim: tuple[float, float],
    phi_center: float,
    theta_center: float,
    n_points: int = 200,
) -> np.ndarray:
    """
    Create the curved boundary of a spherical segment in Hammer projection.
    """
    phi_min, phi_max = phi_lim
    theta_min, theta_max = theta_lim

    boundary_points: list[tuple[float, float]] = []

    phi_bottom = np.linspace(phi_min, phi_max, n_points // 4)
    theta_bottom = np.full_like(phi_bottom, theta_min)
    x_bottom, y_bottom = hammer_projection(
        phi_bottom,
        theta_bottom,
        phi_center,
        theta_center,
    )
    boundary_points.extend(list(zip(x_bottom, y_bottom)))

    theta_right = np.linspace(theta_min, theta_max, n_points // 4)
    phi_right = np.full_like(theta_right, phi_max)
    x_right, y_right = hammer_projection(
        phi_right,
        theta_right,
        phi_center,
        theta_center,
    )
    boundary_points.extend(list(zip(x_right, y_right)))

    phi_top = np.linspace(phi_max, phi_min, n_points // 4)
    theta_top = np.full_like(phi_top, theta_max)
    x_top, y_top = hammer_projection(
        phi_top,
        theta_top,
        phi_center,
        theta_center,
    )
    boundary_points.extend(list(zip(x_top, y_top)))

    theta_left = np.linspace(theta_max, theta_min, n_points // 4)
    phi_left = np.full_like(theta_left, phi_min)
    x_left, y_left = hammer_projection(
        phi_left,
        theta_left,
        phi_center,
        theta_center,
    )
    boundary_points.extend(list(zip(x_left, y_left)))

    return np.array(boundary_points, dtype=float)


def _membrane_to_hammer_dataframe(
    membrane: MembraneMorphometry,
    value_col: str,
    leaflet: str,
) -> pd.DataFrame:
    if leaflet not in {"inner", "outer"}:
        raise ValueError("leaflet must be either 'inner' or 'outer'.")

    membrane_frame = membrane.current_membrane_model.copy()
    membrane_frame = membrane_frame.loc[membrane_frame["leaflet"] == leaflet].copy()
    if membrane_frame.empty:
        raise ValueError(f"No membrane rows are available for leaflet '{leaflet}'.")

    if value_col not in membrane_frame.columns:
        raise ValueError(f"'{value_col}' is not present in the membrane dataframe.")

    x_coord = membrane_frame["x"].to_numpy(dtype=float)
    y_coord = membrane_frame["y"].to_numpy(dtype=float)
    z_coord = membrane_frame["z"].to_numpy(dtype=float)

    rho = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
    theta = np.arctan2(y_coord, x_coord)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.arccos(np.where(rho == 0, np.nan, z_coord / rho))

    return pd.DataFrame(
        {
            "theta": theta,
            "phi": phi,
            "value": membrane_frame[value_col].to_numpy(dtype=float),
        }
    )


def _model_mask_to_hammer_dataframe(model_mask: ModelMask) -> pd.DataFrame:
    mask_frame = model_mask.to_dataframe()
    required_columns = {"theta", "phi"}
    missing_columns = sorted(required_columns - set(mask_frame.columns))
    if missing_columns:
        raise ValueError(
            "ModelMask dataframe is missing required columns: "
            f"{missing_columns}"
        )

    return pd.DataFrame(
        {
            "theta": mask_frame["theta"].to_numpy(dtype=float),
            "phi": mask_frame["phi"].to_numpy(dtype=float),
            "value": np.ones(len(mask_frame), dtype=float),
        }
    )


def _prepare_projection_dataframe(
    dataframe: pd.DataFrame,
    *,
    theta_lim: tuple[float, float] | None,
    phi_lim: tuple[float, float] | None,
) -> pd.DataFrame:
    filtered = dataframe.copy()
    valid_mask = (
        filtered["theta"].notna()
        & filtered["phi"].notna()
        & filtered["value"].notna()
    )
    filtered = filtered.loc[valid_mask].copy()

    if theta_lim is not None:
        filtered = filtered.loc[
            (filtered["theta"] >= theta_lim[0]) & (filtered["theta"] <= theta_lim[1])
        ].copy()
    if phi_lim is not None:
        filtered = filtered.loc[
            (filtered["phi"] >= phi_lim[0]) & (filtered["phi"] <= phi_lim[1])
        ].copy()

    if filtered.empty:
        raise ValueError("No data remain after applying the selected angular limits.")

    return filtered.reset_index(drop=True)


def _value_scaling(
    value_col: str,
    value_array: np.ndarray,
) -> tuple[np.ndarray, str]:
    if value_col == "gaussian_curvature":
        return (
            value_array * 1e6,
            r"Gaussian curvature ($\times 10^{-6}\,\mathrm{\AA}^{-2}$)",
        )

    if value_col == "mean_curvature":
        return (
            value_array * 1e3,
            r"Mean curvature ($\times 10^{-3}\,\mathrm{\AA}^{-1}$)",
        )

    if value_col == "spacing":
        return value_array, "Spacing (A)"

    return value_array, value_col


def _setup_boundary(
    *,
    phi_lim: tuple[float, float] | None,
    theta_lim: tuple[float, float] | None,
) -> tuple[float, float, np.ndarray | None, MplPath | None]:
    phi_center = (phi_lim[0] + phi_lim[1]) / 2 if phi_lim else np.pi
    theta_center = (theta_lim[0] + theta_lim[1]) / 2 if theta_lim else np.pi / 2

    boundary_points = None
    boundary_path = None
    if phi_lim is not None and theta_lim is not None:
        boundary_points = create_segment_boundary(
            phi_lim,
            theta_lim,
            phi_center,
            theta_center,
        )
        boundary_path = MplPath(boundary_points)

    return phi_center, theta_center, boundary_points, boundary_path


def _plot_reference_grid(
    axis,
    *,
    phi_lim: tuple[float, float] | None,
    theta_lim: tuple[float, float] | None,
    phi_center: float,
    theta_center: float,
    n_tick: int,
) -> None:
    def project_phi(phi_value: float) -> tuple[np.ndarray, np.ndarray]:
        theta_line = np.linspace(
            theta_lim[0] if theta_lim else 0,
            theta_lim[1] if theta_lim else np.pi,
            100,
        )
        return hammer_projection(
            np.full_like(theta_line, phi_value),
            theta_line,
            phi_center,
            theta_center,
        )

    def project_theta(theta_value: float) -> tuple[np.ndarray, np.ndarray]:
        phi_line = np.linspace(
            phi_lim[0] if phi_lim else 0,
            phi_lim[1] if phi_lim else 2 * np.pi,
            100,
        )
        return hammer_projection(
            phi_line,
            np.full_like(phi_line, theta_value),
            phi_center,
            theta_center,
        )

    phi_values = (
        np.linspace(phi_lim[0], phi_lim[1], n_tick)
        if phi_lim
        else np.linspace(0, 2 * np.pi, 9)
    )
    theta_values = (
        np.linspace(theta_lim[0], theta_lim[1], n_tick)
        if theta_lim
        else np.linspace(0, np.pi, 6)
    )

    for phi_value in phi_values:
        x_line, y_line = project_phi(phi_value)
        axis.plot(x_line, y_line, color="#000", lw=0.5, alpha=0.3)

    for theta_value in theta_values:
        x_line, y_line = project_theta(theta_value)
        axis.plot(x_line, y_line, color="#000", lw=0.5, alpha=0.3)


def _draw_boundary_ticks(
    axis,
    *,
    phi_lim: tuple[float, float] | None,
    theta_lim: tuple[float, float] | None,
    phi_center: float,
    theta_center: float,
    n_tick: int,
    show_tick_labels: bool,
) -> None:
    if phi_lim is None or theta_lim is None:
        return

    phi_ticks = np.linspace(phi_lim[0], phi_lim[1], n_tick)
    theta_ticks = np.linspace(theta_lim[0], theta_lim[1], n_tick)

    delta_theta = 0.02 * (theta_lim[1] - theta_lim[0])
    delta_phi = 0.02 * (phi_lim[1] - phi_lim[0])

    def add_tick(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label_text: str,
        horizontal_alignment: str,
        vertical_alignment: str,
    ) -> None:
        axis.plot([x0, x1], [y0, y1], color="k", lw=1.0)
        if show_tick_labels:
            dx = x1 - x0
            dy = y1 - y0
            axis.text(
                x1 + 0.6 * dx,
                y1 + 0.6 * dy,
                label_text,
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=12,
            )

    for phi_tick in phi_ticks:
        x0, y0 = hammer_projection(phi_tick, theta_lim[0], phi_center, theta_center)
        x1, y1 = hammer_projection(
            phi_tick,
            theta_lim[0] - delta_theta,
            phi_center,
            theta_center,
        )
        add_tick(
            x0,
            y0,
            x1,
            y1,
            f"{np.degrees(phi_tick):.0f}\N{DEGREE SIGN}",
            "center",
            "top",
        )

    for theta_tick in theta_ticks:
        x0, y0 = hammer_projection(phi_lim[0], theta_tick, phi_center, theta_center)
        x1, y1 = hammer_projection(
            phi_lim[0] - delta_phi,
            theta_tick,
            phi_center,
            theta_center,
        )
        add_tick(
            x0,
            y0,
            x1,
            y1,
            f"{np.degrees(theta_tick):.0f}\N{DEGREE SIGN}",
            "right",
            "center",
        )

        x0r, y0r = hammer_projection(phi_lim[1], theta_tick, phi_center, theta_center)
        x1r, y1r = hammer_projection(
            phi_lim[1] + delta_phi,
            theta_tick,
            phi_center,
            theta_center,
        )
        add_tick(
            x0r,
            y0r,
            x1r,
            y1r,
            f"{np.degrees(theta_tick):.0f}\N{DEGREE SIGN}",
            "left",
            "center",
        )


def _plot_mask_outline(
    axis,
    mask_dataframe: pd.DataFrame,
    *,
    phi_center: float,
    theta_center: float,
    boundary_points: np.ndarray | None,
    boundary_path: MplPath | None,
    color: str,
    linewidth: float,
    alpha: float,
    alpha_shape: float,
) -> None:
    x_data, y_data = hammer_projection(
        mask_dataframe["phi"].to_numpy(dtype=float),
        mask_dataframe["theta"].to_numpy(dtype=float),
        phi_center,
        theta_center,
    )

    try:
        import alphashape
        from shapely.geometry import MultiPolygon, Polygon
    except ImportError:
        alphashape = None

    if alphashape is not None:
        alpha_geometry = alphashape.alphashape(list(zip(x_data, y_data)), alpha=alpha_shape)
        polygons = (
            [alpha_geometry]
            if isinstance(alpha_geometry, Polygon)
            else list(alpha_geometry.geoms)
            if isinstance(alpha_geometry, MultiPolygon)
            else []
        )

        if polygons:
            for polygon in polygons:
                x_coords, y_coords = polygon.exterior.xy
                axis.plot(
                    x_coords,
                    y_coords,
                    color=color,
                    lw=linewidth,
                    alpha=alpha,
                )
            return

        warnings.warn(
            "alphashape returned an empty geometry for the mask overlay; "
            "falling back to projected points.",
            stacklevel=2,
        )

    axis.scatter(
        x_data,
        y_data,
        s=1.0,
        color=color,
        alpha=min(alpha, 0.8),
        linewidths=0,
    )

    if boundary_path is not None and boundary_points is not None:
        axis.plot(boundary_points[:, 0], boundary_points[:, 1], color="black", lw=1.5)


def plot_model_mask_hammer(
    model_mask: ModelMask,
    *,
    title: str = "Model mask footprint",
    theta_lim: tuple[float, float] | None = None,
    phi_lim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 8),
    n_tick: int = 5,
    show_tick_labels: bool = True,
    outline_color: str = "black",
    outline_linewidth: float = 1.2,
    outline_alpha: float = 1.0,
    alpha_shape: float = 30.0,
):
    mask_dataframe = _prepare_projection_dataframe(
        _model_mask_to_hammer_dataframe(model_mask),
        theta_lim=theta_lim,
        phi_lim=phi_lim,
    )

    phi_center, theta_center, boundary_points, boundary_path = _setup_boundary(
        phi_lim=phi_lim,
        theta_lim=theta_lim,
    )

    figure, axis = plt.subplots(figsize=figsize, facecolor="none")
    axis.set_facecolor("none")
    axis.patch.set_alpha(0)

    _plot_mask_outline(
        axis,
        mask_dataframe,
        phi_center=phi_center,
        theta_center=theta_center,
        boundary_points=boundary_points,
        boundary_path=boundary_path,
        color=outline_color,
        linewidth=outline_linewidth,
        alpha=outline_alpha,
        alpha_shape=alpha_shape,
    )

    if boundary_points is not None:
        axis.set_xlim(boundary_points[:, 0].min() - 0.05, boundary_points[:, 0].max() + 0.05)
        axis.set_ylim(boundary_points[:, 1].min() - 0.05, boundary_points[:, 1].max() + 0.05)
    else:
        projected_x, projected_y = hammer_projection(
            mask_dataframe["phi"].to_numpy(dtype=float),
            mask_dataframe["theta"].to_numpy(dtype=float),
            phi_center,
            theta_center,
        )
        axis.set_xlim(projected_x.min() - 0.05, projected_x.max() + 0.05)
        axis.set_ylim(projected_y.min() - 0.05, projected_y.max() + 0.05)

    _plot_reference_grid(
        axis,
        phi_lim=phi_lim,
        theta_lim=theta_lim,
        phi_center=phi_center,
        theta_center=theta_center,
        n_tick=n_tick,
    )
    _draw_boundary_ticks(
        axis,
        phi_lim=phi_lim,
        theta_lim=theta_lim,
        phi_center=phi_center,
        theta_center=theta_center,
        n_tick=n_tick,
        show_tick_labels=show_tick_labels,
    )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=11, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.tight_layout()
    return figure, axis


def plot_membrane_hammer(
    membrane: MembraneMorphometry,
    *,
    value_col: str,
    leaflet: str,
    title: str | None = None,
    theta_lim: tuple[float, float] | None = None,
    phi_lim: tuple[float, float] | None = None,
    n_levels: int = 21,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (10, 8),
    n_tick: int = 5,
    show_tick_labels: bool = True,
    overlay_mask: ModelMask | None = None,
    overlay_color: str = "black",
    overlay_linewidth: float = 1.2,
    overlay_alpha: float = 1.0,
    overlay_alpha_shape: float = 30.0,
):
    if leaflet not in {"inner", "outer"}:
        raise ValueError("leaflet must be either 'inner' or 'outer'.")

    membrane_dataframe = _prepare_projection_dataframe(
        _membrane_to_hammer_dataframe(membrane, value_col=value_col, leaflet=leaflet),
        theta_lim=theta_lim,
        phi_lim=phi_lim,
    )

    phi_center, theta_center, boundary_points, boundary_path = _setup_boundary(
        phi_lim=phi_lim,
        theta_lim=theta_lim,
    )

    projected_x, projected_y = hammer_projection(
        membrane_dataframe["phi"].to_numpy(dtype=float),
        membrane_dataframe["theta"].to_numpy(dtype=float),
        phi_center,
        theta_center,
    )

    scaled_values, colorbar_label = _value_scaling(
        value_col,
        membrane_dataframe["value"].to_numpy(dtype=float),
    )

    figure, axis = plt.subplots(figsize=figsize, facecolor="none")
    axis.set_facecolor("none")
    axis.patch.set_alpha(0)

    pad = 0.15
    x_min = projected_x.min()
    x_max = projected_x.max()
    y_min = projected_y.min()
    y_max = projected_y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= pad * x_range
    x_max += pad * x_range
    y_min -= pad * y_range
    y_max += pad * y_range

    grid_size = 180
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_z = griddata(
        np.column_stack((projected_x, projected_y)),
        scaled_values,
        (grid_x, grid_y),
        method="linear",
    )

    if boundary_path is not None:
        inside = boundary_path.contains_points(
            np.column_stack((grid_x.ravel(), grid_y.ravel()))
        )
        grid_z = np.where(inside.reshape(grid_x.shape), grid_z, np.nan)

    if value_col == "spacing":
        cmap_used = "YlOrBr"
        value_min = np.nanmin(scaled_values)
        value_max = np.nanmax(scaled_values)
        levels = np.linspace(value_min, value_max, n_levels)
    elif value_col == "gaussian_curvature":
        cmap_used = cmap
        value_max = np.nanmax(np.abs(scaled_values))
        levels = np.linspace(-value_max, value_max, n_levels)
    else:
        cmap_used = cmap
        value_max = np.nanmax(np.abs(scaled_values))
        levels = np.linspace(-value_max, value_max, n_levels)

    contour_fill = axis.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=levels,
        cmap=cmap_used,
        extend="both",
        antialiased=True,
    )
    axis.contour(
        grid_x,
        grid_y,
        grid_z,
        levels=levels[::2],
        colors="k",
        linewidths=0.3,
        alpha=0.15,
    )

    if boundary_points is not None:
        axis.plot(boundary_points[:, 0], boundary_points[:, 1], color="black", lw=1.5)
        axis.set_xlim(boundary_points[:, 0].min() - 0.05, boundary_points[:, 0].max() + 0.05)
        axis.set_ylim(boundary_points[:, 1].min() - 0.05, boundary_points[:, 1].max() + 0.05)
    else:
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)

    if overlay_mask is not None:
        overlay_dataframe = _prepare_projection_dataframe(
            _model_mask_to_hammer_dataframe(overlay_mask),
            theta_lim=theta_lim,
            phi_lim=phi_lim,
        )
        _plot_mask_outline(
            axis,
            overlay_dataframe,
            phi_center=phi_center,
            theta_center=theta_center,
            boundary_points=boundary_points,
            boundary_path=boundary_path,
            color=overlay_color,
            linewidth=overlay_linewidth,
            alpha=overlay_alpha,
            alpha_shape=overlay_alpha_shape,
        )

    _plot_reference_grid(
        axis,
        phi_lim=phi_lim,
        theta_lim=theta_lim,
        phi_center=phi_center,
        theta_center=theta_center,
        n_tick=n_tick,
    )
    _draw_boundary_ticks(
        axis,
        phi_lim=phi_lim,
        theta_lim=theta_lim,
        phi_center=phi_center,
        theta_center=theta_center,
        n_tick=n_tick,
        show_tick_labels=show_tick_labels,
    )

    colorbar = figure.colorbar(
        contour_fill,
        ax=axis,
        orientation="vertical",
        pad=0.02,
        shrink=0.8,
    )
    colorbar.set_label(colorbar_label, fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    colorbar.ax.set_facecolor("none")
    colorbar.ax.patch.set_alpha(0)

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title or f"{value_col} ({leaflet})", fontsize=11, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.tight_layout()
    return figure, axis
