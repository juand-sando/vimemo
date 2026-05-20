from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .diagnostic_plotting import DiagnosticPlotting


class MembraneMorphometry(DiagnosticPlotting):
    """
    Derive leaflet annotations, spacing, and curvature measurements from a
    cleaned maxima model.
    """

    REQUIRED_COLUMNS = {"x", "y", "z", "distance", "direction_index"}

    def __init__(
        self,
        maxima_frame: pd.DataFrame,
    ) -> None:
        if not isinstance(maxima_frame, pd.DataFrame):
            raise TypeError("maxima_frame must be a pandas DataFrame.")

        missing_columns = sorted(self.REQUIRED_COLUMNS - set(maxima_frame.columns))
        if missing_columns:
            raise ValueError(
                "maxima_frame is missing required columns: "
                f"{missing_columns}"
            )

        self.current_membrane_model = maxima_frame.copy().reset_index(drop=True)
        self._ensure_annotation_columns()
        self.assign_leaflets_propagated()

    @classmethod
    def from_maxima_model(
        cls,
        maxima_model,
    ) -> "MembraneMorphometry":
        if not hasattr(maxima_model, "current_maxima_model"):
            raise TypeError(
                "maxima_model must expose a current_maxima_model attribute."
            )

        return cls(maxima_model.current_maxima_model)

    def _ensure_annotation_columns(self) -> None:
        row_count = len(self.current_membrane_model)

        string_columns = [
            "leaflet",
            "leaflet_source",
            "pair_status",
        ]
        for column_name in string_columns:
            if column_name not in self.current_membrane_model.columns:
                self.current_membrane_model[column_name] = pd.Series(
                    pd.array([pd.NA] * row_count, dtype="string")
                )

        if "partner_index" not in self.current_membrane_model.columns:
            self.current_membrane_model["partner_index"] = pd.Series(
                pd.array([pd.NA] * row_count, dtype="Int64")
            )

        float_columns = [
            "leaflet_code",
            "leaflet_vote_score",
            "partner_distance",
            "spacing",
            "k1",
            "k2",
            "gaussian_curvature",
            "mean_curvature",
            "nn_distance",
            "mean_neighbor_distance",
            "outlier_score",
            "is_leaflet_outlier",
        ]
        for column_name in float_columns:
            if column_name not in self.current_membrane_model.columns:
                self.current_membrane_model[column_name] = np.nan

    def to_dataframe(self) -> pd.DataFrame:
        return self.current_membrane_model.copy()

    def _get_plot_frame(self) -> pd.DataFrame:
        return self.current_membrane_model

    def _get_plot_frame_for_leaflet(
        self,
        leaflet: str | None,
    ) -> pd.DataFrame:
        if leaflet is None:
            return self.current_membrane_model

        if leaflet not in {"inner", "outer"}:
            raise ValueError("leaflet must be one of: None, 'inner', 'outer'.")

        filtered_frame = self.current_membrane_model.loc[
            self.current_membrane_model["leaflet"] == leaflet
        ].copy()
        if filtered_frame.empty:
            raise ValueError(f"No rows are available for leaflet '{leaflet}'.")

        return filtered_frame

    def static_plot(
        self,
        color_by: str = "intensity",
        marker_size: float = 10,
        alpha: float = 0.7,
        leaflet: str | None = None,
    ) -> None:
        original_frame = self.current_membrane_model
        try:
            self.current_membrane_model = self._get_plot_frame_for_leaflet(leaflet)
            super().static_plot(
                color_by=color_by,
                marker_size=marker_size,
                alpha=alpha,
            )
        finally:
            self.current_membrane_model = original_frame

    def interactive_plot(
        self,
        color_by: str = "intensity",
        marker_size: float = 5,
        opacity: float = 0.8,
        leaflet: str | None = None,
    ) -> None:
        original_frame = self.current_membrane_model
        try:
            self.current_membrane_model = self._get_plot_frame_for_leaflet(leaflet)
            super().interactive_plot(
                color_by=color_by,
                marker_size=marker_size,
                opacity=opacity,
            )
        finally:
            self.current_membrane_model = original_frame

    def _set_leaflet(
        self,
        row_index: int,
        leaflet_code: int,
        source: str,
    ) -> None:
        if leaflet_code not in (-1, 1):
            raise ValueError("leaflet_code must be either -1 or 1.")

        leaflet_label = "inner" if leaflet_code == -1 else "outer"
        self.current_membrane_model.at[row_index, "leaflet"] = leaflet_label
        self.current_membrane_model.at[row_index, "leaflet_code"] = float(leaflet_code)
        self.current_membrane_model.at[row_index, "leaflet_source"] = source

    def _reset_leaflet_assignment_columns(self) -> None:
        row_count = len(self.current_membrane_model)
        self.current_membrane_model["leaflet"] = pd.Series(
            pd.array([pd.NA] * row_count, dtype="string")
        )
        self.current_membrane_model["leaflet_source"] = pd.Series(
            pd.array([pd.NA] * row_count, dtype="string")
        )
        self.current_membrane_model["leaflet_code"] = np.nan
        self.current_membrane_model["leaflet_vote_score"] = np.nan
        self.current_membrane_model["pair_status"] = pd.Series(
            pd.array([pd.NA] * row_count, dtype="string")
        )

    def _reset_spacing_columns(self) -> None:
        row_count = len(self.current_membrane_model)
        self.current_membrane_model["partner_index"] = pd.Series(
            pd.array([pd.NA] * row_count, dtype="Int64")
        )
        self.current_membrane_model["partner_distance"] = np.nan
        self.current_membrane_model["spacing"] = np.nan

    def _reset_morphometry_columns(self) -> None:
        self.current_membrane_model[
            [
                "k1",
                "k2",
                "gaussian_curvature",
                "mean_curvature",
            ]
        ] = np.nan

    def _assign_spacing_by_direction(self) -> None:
        """
        Pair rows strictly by direction_index and compute membrane spacing for
        directions that contain exactly two points without changing leaflet labels.
        """
        self._reset_spacing_columns()

        for _, group in self.current_membrane_model.groupby("direction_index", sort=True):
            row_indices = group.index.to_list()

            if len(row_indices) == 2:
                sorted_group = group.sort_values("distance")
                inner_index, outer_index = sorted_group.index.to_list()

                inner_distance = float(
                    self.current_membrane_model.at[inner_index, "distance"]
                )
                outer_distance = float(
                    self.current_membrane_model.at[outer_index, "distance"]
                )
                spacing = outer_distance - inner_distance

                self.current_membrane_model.at[inner_index, "partner_index"] = outer_index
                self.current_membrane_model.at[outer_index, "partner_index"] = inner_index

                self.current_membrane_model.at[inner_index, "partner_distance"] = outer_distance
                self.current_membrane_model.at[outer_index, "partner_distance"] = inner_distance

                self.current_membrane_model.at[inner_index, "spacing"] = spacing
                self.current_membrane_model.at[outer_index, "spacing"] = spacing

    def assign_leaflets_propagated(
        self,
        neighbor_count: int = 12,
        max_neighbor_distance: float | None = None,
    ) -> None:
        """
        Assign leaflets from exact direction pairs, then propagate labels to
        singleton directions using nearby labelled points.
        """
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be at least 1.")

        self._reset_leaflet_assignment_columns()
        self._reset_spacing_columns()
        self._reset_morphometry_columns()
        self.current_membrane_model[
            [
                "nn_distance",
                "mean_neighbor_distance",
                "outlier_score",
                "is_leaflet_outlier",
            ]
        ] = np.nan

        for _, group in self.current_membrane_model.groupby("direction_index", sort=True):
            row_indices = group.index.to_list()

            if len(row_indices) == 1:
                self.current_membrane_model.at[row_indices[0], "pair_status"] = "unpaired"
                continue

            if len(row_indices) == 2:
                self.current_membrane_model.at[row_indices[0], "pair_status"] = "paired"
                self.current_membrane_model.at[row_indices[1], "pair_status"] = "paired"
                sorted_group = group.sort_values("distance")
                inner_index, outer_index = sorted_group.index.to_list()
                self._set_leaflet(inner_index, -1, "paired")
                self._set_leaflet(outer_index, 1, "paired")
                continue

            for row_index in row_indices:
                self.current_membrane_model.at[row_index, "pair_status"] = "overpopulated"

        labelled_mask = self.current_membrane_model["leaflet_code"].notna()
        if not labelled_mask.any():
            raise ValueError(
                "Leaflet propagation requires at least one direction with exactly two points."
            )

        unlabelled_indices = self.current_membrane_model.index[
            ~labelled_mask
        ].to_list()
        if not unlabelled_indices:
            return

        labelled_coordinates = self.current_membrane_model.loc[
            labelled_mask,
            ["x", "y", "z"],
        ].to_numpy(dtype=float)
        labelled_codes = self.current_membrane_model.loc[
            labelled_mask,
            "leaflet_code",
        ].to_numpy(dtype=float)

        tree = cKDTree(labelled_coordinates)
        query_count = min(neighbor_count, len(labelled_coordinates))

        unlabelled_coordinates = self.current_membrane_model.loc[
            unlabelled_indices,
            ["x", "y", "z"],
        ].to_numpy(dtype=float)
        distances, neighbor_positions = tree.query(unlabelled_coordinates, k=query_count)

        if query_count == 1:
            distances = distances[:, np.newaxis]
            neighbor_positions = neighbor_positions[:, np.newaxis]

        for local_row_position, row_index in enumerate(unlabelled_indices):
            distance_row = np.asarray(distances[local_row_position], dtype=float)
            neighbor_row = np.asarray(neighbor_positions[local_row_position], dtype=int)

            valid_mask = np.isfinite(distance_row)
            if max_neighbor_distance is not None:
                valid_mask &= distance_row <= max_neighbor_distance

            if not valid_mask.any():
                continue

            valid_distances = distance_row[valid_mask]
            valid_codes = labelled_codes[neighbor_row[valid_mask]]

            weights = 1.0 / np.maximum(valid_distances, 1e-12)
            vote_score = float(np.dot(weights, valid_codes) / np.sum(weights))
            propagated_code = 1 if vote_score >= 0.0 else -1

            self._set_leaflet(row_index, propagated_code, "propagated")
            self.current_membrane_model.at[row_index, "leaflet_vote_score"] = vote_score

    def calculate_morphometrics(
        self,
        curvature_neighbor_radius_ratio: float = 20.0,
        min_neighbors: int = 6,
    ) -> None:
        """
        Compute spacing and curvature measurements after leaflet cleanup.
        """
        outlier_mask = self.current_membrane_model["is_leaflet_outlier"] == 1.0
        remaining_outlier_count = int(np.nansum(outlier_mask.to_numpy(dtype=float)))
        if remaining_outlier_count > 0:
            print(
                f"Warning: {remaining_outlier_count} points are still flagged as outliers. "
                "Consider plotting them and removing them before calculating morphometrics."
            )

        self._assign_spacing_by_direction()
        print("Spacing between leaflets calculated.")
        self._reset_morphometry_columns()
        self.compute_principal_curvatures(
            leaflet="inner",
            neighbor_radius_ratio=curvature_neighbor_radius_ratio,
            min_neighbors=min_neighbors,
        )
        print("Inner leaflet curvature calculated.")
        self.compute_principal_curvatures(
            leaflet="outer",
            neighbor_radius_ratio=curvature_neighbor_radius_ratio,
            min_neighbors=min_neighbors,
        )
        print("Outer leaflet curvature calculated.")

    def compute_principal_curvatures(
        self,
        leaflet: str | None = None,
        neighbor_radius_ratio: float = 20.0,
        min_neighbors: int = 6,
    ) -> None:
        """
        Estimate principal curvatures on a leaflet-by-leaflet basis using a
        PCA-aligned local quadratic surface fit.
        """
        if leaflet not in (None, "inner", "outer"):
            raise ValueError("leaflet must be one of: None, 'inner', 'outer'.")

        if neighbor_radius_ratio <= 0.0:
            raise ValueError("neighbor_radius_ratio must be positive.")

        if leaflet is None:
            leaflet_labels = [
                leaflet_label
                for leaflet_label in ("inner", "outer")
                if (self.current_membrane_model["leaflet"] == leaflet_label).any()
            ]
        else:
            leaflet_labels = [leaflet]

        for leaflet_label in leaflet_labels:
            leaflet_mask = self.current_membrane_model["leaflet"] == leaflet_label
            leaflet_frame = self.current_membrane_model.loc[leaflet_mask]

            if leaflet_frame.empty:
                continue

            self.current_membrane_model.loc[
                leaflet_mask,
                ["k1", "k2", "gaussian_curvature", "mean_curvature"],
            ] = np.nan

            coordinates = leaflet_frame[["x", "y", "z"]].to_numpy(dtype=float)
            tree = cKDTree(coordinates)

            if len(coordinates) < 2:
                continue

            nearest_neighbor_distances, _ = tree.query(coordinates, k=2)
            average_nearest_neighbor_distance = float(
                np.mean(nearest_neighbor_distances[:, 1])
            )
            if (
                not np.isfinite(average_nearest_neighbor_distance)
                or average_nearest_neighbor_distance <= 0.0
            ):
                continue
            neighbor_radius = neighbor_radius_ratio * average_nearest_neighbor_distance

            for local_row_position, row_index in enumerate(leaflet_frame.index):
                point = coordinates[local_row_position]
                neighbor_row = np.asarray(
                    tree.query_ball_point(point, r=neighbor_radius),
                    dtype=int,
                )

                if neighbor_row.size < min_neighbors:
                    continue

                neighborhood = coordinates[neighbor_row]
                centered_neighborhood = neighborhood - point

                covariance = centered_neighborhood.T @ centered_neighborhood
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                order = np.argsort(eigenvalues)

                normal = eigenvectors[:, order[0]]
                point_norm = np.linalg.norm(point)
                if point_norm > 0.0 and np.dot(normal, point / point_norm) > 0.0:
                    normal = -normal
                normal /= np.linalg.norm(normal)

                tangent_u = eigenvectors[:, order[2]]
                tangent_u /= np.linalg.norm(tangent_u)
                tangent_v = np.cross(normal, tangent_u)
                tangent_v_norm = np.linalg.norm(tangent_v)
                if tangent_v_norm <= 1e-12:
                    continue
                tangent_v /= tangent_v_norm

                local_basis = np.column_stack((tangent_u, tangent_v, normal))
                local_coordinates = centered_neighborhood @ local_basis

                u_coords = local_coordinates[:, 0]
                v_coords = local_coordinates[:, 1]
                w_coords = local_coordinates[:, 2]

                design_matrix = np.column_stack(
                    (
                        0.5 * u_coords**2,
                        u_coords * v_coords,
                        0.5 * v_coords**2,
                        u_coords,
                        v_coords,
                        np.ones_like(u_coords),
                    )
                )

                try:
                    coefficients, _, _, _ = np.linalg.lstsq(
                        design_matrix,
                        w_coords,
                        rcond=None,
                    )
                except np.linalg.LinAlgError:
                    continue

                curvature_uu, curvature_uv, curvature_vv, slope_u, slope_v, _ = coefficients

                first_fundamental_form = np.array(
                    [
                        [1.0 + slope_u**2, slope_u * slope_v],
                        [slope_u * slope_v, 1.0 + slope_v**2],
                    ],
                    dtype=float,
                )
                normal_scale = np.sqrt(1.0 + slope_u**2 + slope_v**2)
                second_fundamental_form = (
                    np.array(
                        [
                            [curvature_uu, curvature_uv],
                            [curvature_uv, curvature_vv],
                        ],
                        dtype=float,
                    )
                    / normal_scale
                )

                try:
                    shape_operator = np.linalg.solve(
                        first_fundamental_form,
                        second_fundamental_form,
                    )
                except np.linalg.LinAlgError:
                    continue

                principal_curvatures = np.linalg.eigvalsh(shape_operator)
                k1 = float(principal_curvatures[0])
                k2 = float(principal_curvatures[1])

                self.current_membrane_model.at[row_index, "k1"] = k1
                self.current_membrane_model.at[row_index, "k2"] = k2
                self.current_membrane_model.at[row_index, "gaussian_curvature"] = k1 * k2
                self.current_membrane_model.at[row_index, "mean_curvature"] = 0.5 * (
                    k1 + k2
                )

    def diagnose_leaflet_outliers(
        self,
        leaflet: str | None = None,
        neighbor_count: int = 8,
        sd_multiplier: float = 3.0,
    ) -> None:
        """
        Diagnose spatial outliers within each leaflet using same-leaflet
        nearest-neighbor distances and a mean/standard-deviation threshold.
        """
        if leaflet not in (None, "inner", "outer"):
            raise ValueError("leaflet must be one of: None, 'inner', 'outer'.")

        if neighbor_count < 1:
            raise ValueError("neighbor_count must be at least 1.")

        if sd_multiplier < 0.0:
            raise ValueError("sd_multiplier must be non-negative.")

        if leaflet is None:
            leaflet_labels = [
                leaflet_label
                for leaflet_label in ("inner", "outer")
                if (self.current_membrane_model["leaflet"] == leaflet_label).any()
            ]
        else:
            leaflet_labels = [leaflet]

        if not leaflet_labels:
            raise ValueError(
                "No leaflet-labelled points are available for outlier diagnosis."
            )

        target_mask = self.current_membrane_model["leaflet"].isin(leaflet_labels)
        self.current_membrane_model.loc[
            target_mask,
            [
                "nn_distance",
                "mean_neighbor_distance",
                "outlier_score",
                "is_leaflet_outlier",
            ],
        ] = np.nan

        for leaflet_label in leaflet_labels:
            leaflet_mask = self.current_membrane_model["leaflet"] == leaflet_label
            leaflet_frame = self.current_membrane_model.loc[leaflet_mask]
            if leaflet_frame.empty:
                continue

            coordinates = leaflet_frame[["x", "y", "z"]].to_numpy(dtype=float)
            point_count = len(coordinates)

            if point_count < 2:
                self.current_membrane_model.loc[
                    leaflet_frame.index,
                    "is_leaflet_outlier",
                ] = 0.0
                continue

            tree = cKDTree(coordinates)
            query_count = min(neighbor_count + 1, point_count)
            distances, _ = tree.query(coordinates, k=query_count)

            neighbor_distances = np.asarray(distances[:, 1:], dtype=float)
            nn_distance = neighbor_distances[:, 0]
            mean_neighbor_distance = np.nanmean(neighbor_distances, axis=1)

            finite_mask = np.isfinite(mean_neighbor_distance)
            if not finite_mask.any():
                self.current_membrane_model.loc[
                    leaflet_frame.index,
                    "is_leaflet_outlier",
                ] = 0.0
                continue

            baseline = mean_neighbor_distance[finite_mask]
            mean_distance = float(np.mean(baseline))
            standard_deviation = float(np.std(baseline))

            outlier_score = np.full(point_count, np.nan, dtype=float)
            if standard_deviation > 0.0:
                outlier_score[finite_mask] = (
                    mean_neighbor_distance[finite_mask] - mean_distance
                ) / standard_deviation
                outlier_threshold = (
                    mean_distance + sd_multiplier * standard_deviation
                )
            else:
                outlier_score[finite_mask] = (
                    mean_neighbor_distance[finite_mask] - mean_distance
                )
                outlier_threshold = mean_distance

            is_leaflet_outlier = np.zeros(point_count, dtype=float)
            is_leaflet_outlier[finite_mask] = (
                mean_neighbor_distance[finite_mask] > outlier_threshold
            ).astype(float)

            self.current_membrane_model.loc[leaflet_frame.index, "nn_distance"] = nn_distance
            self.current_membrane_model.loc[
                leaflet_frame.index,
                "mean_neighbor_distance",
            ] = mean_neighbor_distance
            self.current_membrane_model.loc[
                leaflet_frame.index,
                "outlier_score",
            ] = outlier_score
            self.current_membrane_model.loc[
                leaflet_frame.index,
                "is_leaflet_outlier",
            ] = is_leaflet_outlier

            outlier_count = int(np.nansum(is_leaflet_outlier))
            print(
                f"Leaflet '{leaflet_label}': {outlier_count} outliers out of {point_count} points. "
                "Plot 'is_leaflet_outlier' or 'outlier_score' to verify them."
            )

    def eliminate_outliers(self) -> None:
        """
        Remove points currently flagged as leaflet outliers.
        """
        if "is_leaflet_outlier" not in self.current_membrane_model.columns:
            raise ValueError(
                "Run diagnose_leaflet_outliers() before eliminate_outliers()."
            )

        if self.current_membrane_model["is_leaflet_outlier"].isna().all():
            raise ValueError(
                "Run diagnose_leaflet_outliers() before eliminate_outliers()."
            )

        outlier_mask = self.current_membrane_model["is_leaflet_outlier"] == 1.0
        removed_count = int(np.nansum(outlier_mask.to_numpy(dtype=float)))
        if removed_count == 0:
            print("No outliers are currently flagged, so no rows were removed.")
            return

        self.current_membrane_model = (
            self.current_membrane_model.loc[~outlier_mask]
            .copy()
            .reset_index(drop=True)
        )
        print(f"Removed {removed_count} outliers from the membrane model.")
