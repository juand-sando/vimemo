from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


class DiagnosticPlotting:
    """
    Shared diagnostic point-cloud plotting for dataframe-backed models.

    Only numeric columns are allowed for coloring, and they are always
    rendered using a continuous color scale.
    """

    def _get_plot_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare_plot_frame(
        self,
        color_by: str,
    ) -> pd.DataFrame:
        plot_frame = self._get_plot_frame()
        if plot_frame.empty:
            raise ValueError("The plot frame is empty, so nothing can be plotted.")

        required_columns = {"x", "y", "z", color_by}
        missing_columns = sorted(required_columns - set(plot_frame.columns))
        if missing_columns:
            raise ValueError(
                "The plot frame is missing required columns: "
                f"{missing_columns}"
            )

        if not pd.api.types.is_numeric_dtype(plot_frame[color_by]):
            raise TypeError(
                f"'{color_by}' must be numeric for diagnostic plotting."
            )

        valid_mask = plot_frame[["x", "y", "z", color_by]].notna().all(axis=1)
        filtered_frame = plot_frame.loc[valid_mask].copy().reset_index(drop=True)
        if filtered_frame.empty:
            raise ValueError(
                "No rows contain finite x, y, z, and color_by values for plotting."
            )

        return filtered_frame

    def static_plot(
        self,
        color_by: str = "intensity",
        marker_size: float = 10,
        alpha: float = 0.7,
    ) -> None:
        plot_frame = self._prepare_plot_frame(color_by=color_by)

        x_vals = plot_frame["x"].to_numpy(dtype=float)
        y_vals = plot_frame["y"].to_numpy(dtype=float)
        z_vals = plot_frame["z"].to_numpy(dtype=float)
        color_values = plot_frame[color_by].to_numpy(dtype=float)

        figure = plt.figure(figsize=(10, 8))
        axis = figure.add_subplot(111, projection="3d")

        scatter = axis.scatter(
            x_vals,
            y_vals,
            z_vals,
            c=color_values,
            cmap="viridis",
            s=marker_size,
            alpha=alpha,
        )

        colorbar = plt.colorbar(scatter, ax=axis, shrink=0.6)
        colorbar.set_label(color_by)

        axis.set_title(f"{self.__class__.__name__} Colored by {color_by}")
        axis.set_xlabel("X (A)")
        axis.set_ylabel("Y (A)")
        axis.set_zlabel("Z (A)")

        plt.tight_layout()
        plt.show()

    def interactive_plot(
        self,
        color_by: str = "intensity",
        marker_size: float = 5,
        opacity: float = 0.8,
    ) -> None:
        plot_frame = self._prepare_plot_frame(color_by=color_by)

        x_vals = plot_frame["x"].to_numpy(dtype=float)
        y_vals = plot_frame["y"].to_numpy(dtype=float)
        z_vals = plot_frame["z"].to_numpy(dtype=float)
        color_values = plot_frame[color_by].to_numpy(dtype=float)

        traces = [
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color_values,
                    colorscale="Viridis",
                    colorbar=dict(title=color_by),
                    opacity=opacity,
                ),
                name=color_by,
            )
        ]

        layout = go.Layout(
            title=f"{self.__class__.__name__} Colored by {color_by}",
            scene=dict(
                xaxis_title="X (A)",
                yaxis_title="Y (A)",
                zaxis_title="Z (A)",
            ),
            height=700,
        )

        figure = go.Figure(data=traces, layout=layout)
        figure.show()
