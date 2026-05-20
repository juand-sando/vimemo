from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from .diagnostic_plotting import DiagnosticPlotting


class MaximaModel(DiagnosticPlotting):
    """
    Identify intensity peaks from surveyed density profiles and manage a
    working maxima model.
    """

    def __init__(
        self,
        survey,
        peak_filter: float = 0.009,
    ) -> None:
        if survey.raw_density_data is None:
            raise ValueError(
                "The CryoEMSurvey object does not contain raw_density_data. "
                "Run calculate_raw_density_data() first."
            )

        self.survey = survey
        self.raw_density_data = survey.raw_density_data
        self.peak_filter = peak_filter

        self.raw_peak_data_array = self._build_peak_data_array()
        self.current_maxima_model = self._build_current_maxima_model()

    def _build_peak_data_array(self) -> np.ndarray:
        """
        Detect peaks in each surveyed direction and combine them into one array.
        """
        peak_rows: list[list[float]] = []

        for direction_index in sorted(self.raw_density_data):
            unit_vector_angst = np.asarray(
                self.raw_density_data[direction_index]["unit_vector_angst"],
                dtype=float,
            )
            survey_data = np.asarray(
                self.raw_density_data[direction_index]["survey_data"],
                dtype=float,
            )

            if survey_data.size == 0:
                continue

            distances = survey_data[:, 0]
            intensities = survey_data[:, 1]

            peak_indices, _ = find_peaks(
                intensities,
                height=self.peak_filter,
                distance=4,
                width=2,
            )

            for peak_index in peak_indices:
                distance = float(distances[peak_index])
                intensity = float(intensities[peak_index])

                x_coord, y_coord, z_coord = unit_vector_angst * distance

                peak_rows.append(
                    [
                        x_coord,
                        y_coord,
                        z_coord,
                        intensity,
                        distance,
                        direction_index,
                    ]
                )

        if not peak_rows:
            return np.empty((0, 6), dtype=float)

        return np.array(peak_rows, dtype=float)

    def _build_current_maxima_model(self) -> pd.DataFrame:
        """
        Build the working maxima model DataFrame from raw_peak_data_array.
        """
        return pd.DataFrame(
            self.raw_peak_data_array,
            columns=[
                "x",
                "y",
                "z",
                "intensity",
                "distance",
                "direction_index",
            ],
        )

    def recover_initial_maxima_model(self) -> None:
        """
        Restore current_maxima_model to the original model built at initialization.
        """
        self.current_maxima_model = (
            self._build_current_maxima_model().reset_index(drop=True)
        )

    def initial_k_clustering(
        self,
        amount_of_clusters: int,
    ) -> None:
        """
        Run the initial K-means clustering on current_maxima_model.
        """
        if self.current_maxima_model.empty:
            raise ValueError(
                "current_maxima_model is empty, so clustering cannot be performed."
            )

        if len(self.current_maxima_model) < amount_of_clusters:
            raise ValueError(
                "The number of detected peaks is smaller than amount_of_clusters."
            )

        intensity_values = self.current_maxima_model["intensity"].to_numpy(dtype=float)
        distance_values = self.current_maxima_model["distance"].to_numpy(dtype=float)

        max_intensity = np.max(intensity_values)
        max_distance = np.max(distance_values)

        if max_intensity == 0:
            raise ValueError(
                "Maximum intensity is zero, so the clustering weight cannot be calculated."
            )

        density_clustering_weight = max_distance / max_intensity * 100
        weighted_intensity_values = intensity_values * density_clustering_weight

        print(density_clustering_weight)

        clustering_input = np.column_stack(
            (
                weighted_intensity_values,
                distance_values,
            )
        )

        kmeans = KMeans(
            n_clusters=amount_of_clusters,
            random_state=0,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(clustering_input)

        self.current_maxima_model["cluster"] = cluster_labels

    def select_cluster(
        self,
        clusters: int | list[int] | tuple[int, ...] | np.ndarray,
    ) -> None:
        """
        Select one or more clusters from current_maxima_model and update
        the working model.
        """
        if self.current_maxima_model.empty:
            raise ValueError(
                "current_maxima_model is empty, so no cluster can be selected."
            )

        if "cluster" not in self.current_maxima_model.columns:
            raise ValueError(
                "The 'cluster' column is not present in current_maxima_model. "
                "Run initial_k_clustering(...) or voxel_connection(...) first."
            )

        if isinstance(clusters, (int, np.integer)):
            selected_clusters = [int(clusters)]
        else:
            selected_clusters = [int(cluster) for cluster in clusters]

        available_clusters = self.current_maxima_model["cluster"].unique().tolist()
        missing_clusters = [
            cluster
            for cluster in selected_clusters
            if cluster not in available_clusters
        ]

        if missing_clusters:
            raise ValueError(
                f"Clusters {missing_clusters} are not present in current_maxima_model. "
                f"Available clusters are: {sorted(available_clusters)}"
            )

        self.current_maxima_model = (
            self.current_maxima_model[
                self.current_maxima_model["cluster"].isin(selected_clusters)
            ]
            .copy()
            .reset_index(drop=True)
        )

    def voxel_connection(
        self,
        voxel_size: float,
        expected_structure_count: int,
        min_neighbor_voxel_count: int = 2,
        connectivity: str = "26",
    ) -> None:
        """
        Reassign current_maxima_model clusters using voxel-based connected
        components.
        """
        if self.current_maxima_model.empty:
            raise ValueError(
                "current_maxima_model is empty, so voxel connection cannot be performed."
            )

        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")

        if expected_structure_count < 1:
            raise ValueError("expected_structure_count must be at least 1.")

        point_matrix = self.current_maxima_model.to_numpy(dtype=float)
        spatial_coordinates = point_matrix[:, 0:3]

        voxel_index_matrix = np.floor(spatial_coordinates / voxel_size).astype(np.int64)

        unique_voxel_indices, point_to_voxel_index = np.unique(
            voxel_index_matrix,
            axis=0,
            return_inverse=True,
        )

        occupied_voxel_count = unique_voxel_indices.shape[0]
        point_labels = np.full(point_matrix.shape[0], -1, dtype=int)

        if occupied_voxel_count < 2:
            self.current_maxima_model["cluster"] = point_labels
            return

        voxel_center_matrix = unique_voxel_indices.astype(np.float64) * voxel_size
        voxel_tree = cKDTree(voxel_center_matrix)

        if connectivity == "6":
            neighbor_radius = 1.01 * voxel_size
        elif connectivity == "18":
            neighbor_radius = 1.01 * np.sqrt(2.0) * voxel_size
        elif connectivity == "26":
            neighbor_radius = 1.01 * np.sqrt(3.0) * voxel_size
        else:
            raise ValueError('connectivity must be one of: "6", "18", "26".')

        neighboring_voxel_pairs = np.array(
            list(voxel_tree.query_pairs(r=neighbor_radius)),
            dtype=np.int64,
        )

        if neighboring_voxel_pairs.size == 0:
            self.current_maxima_model["cluster"] = point_labels
            return

        voxel_neighbor_count = np.zeros(occupied_voxel_count, dtype=np.int64)
        np.add.at(voxel_neighbor_count, neighboring_voxel_pairs[:, 0], 1)
        np.add.at(voxel_neighbor_count, neighboring_voxel_pairs[:, 1], 1)

        supported_voxel_mask = voxel_neighbor_count >= min_neighbor_voxel_count
        supported_voxel_indices = np.where(supported_voxel_mask)[0]

        if supported_voxel_indices.size < 1:
            self.current_maxima_model["cluster"] = point_labels
            return

        old_voxel_index_to_supported_index = -np.ones(
            occupied_voxel_count,
            dtype=np.int64,
        )
        old_voxel_index_to_supported_index[supported_voxel_indices] = np.arange(
            supported_voxel_indices.size
        )

        valid_pair_mask = (
            supported_voxel_mask[neighboring_voxel_pairs[:, 0]]
            & supported_voxel_mask[neighboring_voxel_pairs[:, 1]]
        )
        supported_voxel_pairs = neighboring_voxel_pairs[valid_pair_mask]

        if supported_voxel_pairs.size == 0:
            self.current_maxima_model["cluster"] = point_labels
            return

        supported_voxel_pairs = old_voxel_index_to_supported_index[
            supported_voxel_pairs
        ]

        row_indices = np.concatenate(
            [supported_voxel_pairs[:, 0], supported_voxel_pairs[:, 1]]
        )
        column_indices = np.concatenate(
            [supported_voxel_pairs[:, 1], supported_voxel_pairs[:, 0]]
        )
        edge_values = np.ones(row_indices.shape[0], dtype=np.uint8)

        voxel_graph = coo_matrix(
            (edge_values, (row_indices, column_indices)),
            shape=(supported_voxel_indices.size, supported_voxel_indices.size),
        )

        _, supported_voxel_component_labels = connected_components(
            voxel_graph,
            directed=False,
        )

        component_sizes = np.bincount(supported_voxel_component_labels)
        component_order_descending = np.argsort(component_sizes)[::-1]
        kept_component_indices = component_order_descending[
            :expected_structure_count
        ]

        voxel_component_labels = np.full(occupied_voxel_count, -1, dtype=int)
        voxel_component_labels[supported_voxel_indices] = (
            supported_voxel_component_labels
        )

        point_component_labels = voxel_component_labels[point_to_voxel_index]

        for new_cluster_label, component_index in enumerate(kept_component_indices):
            point_labels[point_component_labels == component_index] = new_cluster_label

        self.current_maxima_model["cluster"] = point_labels

    def cleanup_per_direction(
        self,
        number_of_neighbors: int = 5,
        epsilon: float = 1e-8,
    ) -> None:
        """
        Clean current_maxima_model by enforcing at most two points per direction.

        Rules
        -----
        - Directions with exactly 2 points are treated as trusted and kept unchanged.
        - Directions with 1 point are kept unchanged, but reported.
        - Directions with more than 2 points are scored against the trusted points,
          and only the best 2 points are kept.
        """
        if self.current_maxima_model.empty:
            raise ValueError(
                "current_maxima_model is empty, so cleanup cannot be performed."
            )

        column_names = list(self.current_maxima_model.columns)

        if "direction_index" not in column_names:
            raise ValueError(
                "current_maxima_model must contain a 'direction_index' column."
            )

        point_matrix = self.current_maxima_model.to_numpy(copy=True)

        x_index = column_names.index("x")
        y_index = column_names.index("y")
        z_index = column_names.index("z")
        direction_index_column = column_names.index("direction_index")

        direction_indices = point_matrix[:, direction_index_column]
        unique_directions, counts = np.unique(direction_indices, return_counts=True)

        trusted_direction_indices = unique_directions[counts == 2]
        singleton_direction_indices = unique_directions[counts == 1]
        crowded_direction_indices = unique_directions[counts > 2]

        print(f"Directions with more than 2 points: {len(crowded_direction_indices)}")
        print(f"Directions with 1 point are kept: {len(singleton_direction_indices)}")

        if len(trusted_direction_indices) == 0:
            raise ValueError(
                "No trusted directions with exactly 2 points were found, "
                "so cleanup cannot be performed."
            )

        kept_row_indices: list[int] = []

        trusted_direction_mask = np.isin(direction_indices, trusted_direction_indices)
        trusted_row_indices = np.flatnonzero(trusted_direction_mask)
        kept_row_indices.extend(trusted_row_indices.tolist())

        singleton_direction_mask = np.isin(
            direction_indices,
            singleton_direction_indices,
        )
        singleton_row_indices = np.flatnonzero(singleton_direction_mask)
        kept_row_indices.extend(singleton_row_indices.tolist())

        trusted_points = point_matrix[trusted_row_indices]
        trusted_coordinates = trusted_points[:, [x_index, y_index, z_index]]

        trusted_tree = cKDTree(trusted_coordinates)
        effective_number_of_neighbors = min(
            number_of_neighbors,
            len(trusted_coordinates),
        )

        for direction_index in crowded_direction_indices:
            row_indices = np.flatnonzero(direction_indices == direction_index)
            candidate_points = point_matrix[row_indices]
            candidate_coordinates = candidate_points[:, [x_index, y_index, z_index]]

            distances, _ = trusted_tree.query(
                candidate_coordinates,
                k=effective_number_of_neighbors,
            )

            if effective_number_of_neighbors == 1:
                distances = distances[:, np.newaxis]

            scores = np.sum(1.0 / (distances + epsilon), axis=1)

            best_two_local_positions = np.argpartition(scores, -2)[-2:]
            best_two_row_indices = row_indices[best_two_local_positions]

            kept_row_indices.extend(best_two_row_indices.tolist())

        kept_row_indices_array = np.array(sorted(kept_row_indices), dtype=int)
        cleaned_matrix = point_matrix[kept_row_indices_array]

        self.current_maxima_model = pd.DataFrame(
            cleaned_matrix,
            columns=column_names,
        ).reset_index(drop=True)

    def _get_plot_frame(self) -> pd.DataFrame:
        return self.current_maxima_model
