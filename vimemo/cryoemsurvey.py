from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import uniform_filter


class CryoEMSurvey:
    """
    Survey a cryo-EM density map along radial directions from the map center.

    Parameters
    ----------
    cryo_em_map
        3D cryo-EM density map as a NumPy array.
    angspix
        Pixel size in Angstroms per pixel.
    survey_radius_min_angst
        Minimum survey radius in Angstroms.
    survey_radius_angst
        Maximum survey radius in Angstroms.
    grid_theta_min_deg, grid_theta_max_deg, grid_phi_min_deg, grid_phi_max_deg
        Optional angular limits for the survey grid. If not provided, they must
        later be set directly with `set_grid_limits(...)` or computed from
        vectors with `calculate_grid_limits(...)`.
    """

    def __init__(
        self,
        cryo_em_map: np.ndarray,
        angspix: float,
        survey_radius_min_angst: float,
        survey_radius_angst: float,
        grid_theta_min_deg: float | None = None,
        grid_theta_max_deg: float | None = None,
        grid_phi_min_deg: float | None = None,
        grid_phi_max_deg: float | None = None,
    ) -> None:
        self.cryo_em_map = np.asarray(cryo_em_map, dtype=float)
        self.angspix = float(angspix)
        self.survey_radius_min_angst = float(survey_radius_min_angst)
        self.survey_radius_angst = float(survey_radius_angst)

        self.map_dims_pix = np.array(self.cryo_em_map.shape, dtype=int)
        self.center_pos_pix = self.map_dims_pix // 2
        self.center_pos_angst = self.center_pos_pix * self.angspix

        self.grid_theta_min_deg: float | None = grid_theta_min_deg
        self.grid_theta_max_deg: float | None = grid_theta_max_deg
        self.grid_phi_min_deg: float | None = grid_phi_min_deg
        self.grid_phi_max_deg: float | None = grid_phi_max_deg

        self.direction_vectors_angst: np.ndarray | None = None
        self.raw_density_data: dict[int, dict[str, np.ndarray]] | None = None
        self.smoothed_cryo_em_map: np.ndarray | None = None

        provided_grid_limit_count = sum(
            value is not None
            for value in [
                self.grid_theta_min_deg,
                self.grid_theta_max_deg,
                self.grid_phi_min_deg,
                self.grid_phi_max_deg,
            ]
        )

        if provided_grid_limit_count not in (0, 4):
            raise ValueError(
                "You must provide either all four grid limits or none of them: "
                "grid_theta_min_deg, grid_theta_max_deg, "
                "grid_phi_min_deg, grid_phi_max_deg."
            )

        if provided_grid_limit_count == 0:
            warnings.warn(
                "Grid limits have not been provided. Direction vectors cannot be "
                "calculated until grid limits are defined. Use either "
                "set_grid_limits(...) or calculate_grid_limits(...).",
                stacklevel=2,
            )

    def prepare_smoothed_map(
        self,
        window_size: int = 1,
    ) -> None:
        """
        Precompute a uniformly smoothed cryo-EM map.

        Parameters
        ----------
        window_size
            Number of pixels included on each side of the central voxel.
            A value of 1 corresponds to a 3 x 3 x 3 smoothing window.
        """
        if window_size < 0:
            raise ValueError("window_size must be non-negative.")

        filter_size = 2 * window_size + 1

        self.smoothed_cryo_em_map = uniform_filter(
            self.cryo_em_map,
            size=filter_size,
            mode="nearest",
        )

    def set_grid_limits(
        self,
        theta_min_deg: float,
        theta_max_deg: float,
        phi_min_deg: float,
        phi_max_deg: float,
    ) -> None:
        """
        Store angular grid limits on the instance.

        This method overwrites any previously stored grid limits and clears any
        previously stored direction vectors and raw density data.
        """
        self.grid_theta_min_deg = float(theta_min_deg)
        self.grid_theta_max_deg = float(theta_max_deg)
        self.grid_phi_min_deg = float(phi_min_deg)
        self.grid_phi_max_deg = float(phi_max_deg)

        self.direction_vectors_angst = None
        self.raw_density_data = None

        warnings.warn(
            "Grid limits have been updated to: "
            f"theta = {self.grid_theta_min_deg:.2f} --> {self.grid_theta_max_deg:.2f} deg, "
            f"phi = {self.grid_phi_min_deg:.2f} --> {self.grid_phi_max_deg:.2f} deg. "
            "Stored direction vectors and raw density data have been cleared.",
            stacklevel=2,
        )

    def calculate_grid_limits(
        self,
        vectors: np.ndarray,
        vector_units: str,
    ) -> tuple[float, float, float, float]:
        """
        Calculate angular grid limits from input vectors and store them.

        Parameters
        ----------
        vectors
            Array of shape (N, 3) containing vectors referenced to the map
            origin (0, 0, 0).
        vector_units
            Units of the vectors. Must be either 'pixels' or 'angstroms'.

        Returns
        -------
        tuple[float, float, float, float]
            (theta_min_deg, theta_max_deg, phi_min_deg, phi_max_deg)
        """
        vectors = np.asarray(vectors, dtype=float)

        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError(
                "vectors must be a 2D NumPy array with shape (N, 3)."
            )

        if vector_units not in ("pixels", "angstroms"):
            raise ValueError(
                "vector_units must be either 'pixels' or 'angstroms'."
            )

        if vector_units == "pixels":
            vectors_angst = vectors * self.angspix
        else:
            vectors_angst = vectors

        theta_values_deg = []
        phi_values_deg = []

        for vector_angst in vectors_angst:
            x_coord_angst, y_coord_angst, z_coord_angst = vector_angst
            vector_length_angst = np.linalg.norm(vector_angst)

            if vector_length_angst == 0:
                raise ValueError(
                    "At least one input vector has zero length, which is not allowed."
                )

            theta_rad = np.arctan2(y_coord_angst, x_coord_angst)
            phi_rad = np.arccos(
                np.clip(z_coord_angst / vector_length_angst, -1.0, 1.0)
            )

            theta_values_deg.append(np.degrees(theta_rad))
            phi_values_deg.append(np.degrees(phi_rad))

        theta_min_deg = float(min(theta_values_deg))
        theta_max_deg = float(max(theta_values_deg))
        phi_min_deg = float(min(phi_values_deg))
        phi_max_deg = float(max(phi_values_deg))

        self.set_grid_limits(
            theta_min_deg=theta_min_deg,
            theta_max_deg=theta_max_deg,
            phi_min_deg=phi_min_deg,
            phi_max_deg=phi_max_deg,
        )

        return (
            theta_min_deg,
            theta_max_deg,
            phi_min_deg,
            phi_max_deg,
        )

    def calculate_direction_vectors_angst(
        self,
        angle_step_deg: float = 0.2,
    ) -> np.ndarray:
        """
        Calculate and store unit direction vectors from the stored grid limits.

        Parameters
        ----------
        angle_step_deg
            Angular step size in degrees.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) containing unit direction vectors.
        """
        if any(
            value is None
            for value in [
                self.grid_theta_min_deg,
                self.grid_theta_max_deg,
                self.grid_phi_min_deg,
                self.grid_phi_max_deg,
            ]
        ):
            warnings.warn(
                "Direction vectors cannot be calculated because grid limits are "
                "not defined. Use set_grid_limits(...) or "
                "calculate_grid_limits(...) first.",
                stacklevel=2,
            )
            raise ValueError("Grid limits are not defined.")

        theta_range_rad = np.radians(
            np.arange(
                self.grid_theta_min_deg,
                self.grid_theta_max_deg + angle_step_deg,
                angle_step_deg,
            )
        )
        phi_range_rad = np.radians(
            np.arange(
                self.grid_phi_min_deg,
                self.grid_phi_max_deg + angle_step_deg,
                angle_step_deg,
            )
        )

        direction_vectors_angst = []

        for theta_rad in theta_range_rad:
            for phi_rad in phi_range_rad:
                x_coord = np.sin(phi_rad) * np.cos(theta_rad)
                y_coord = np.sin(phi_rad) * np.sin(theta_rad)
                z_coord = np.cos(phi_rad)

                unit_vector_angst = np.array(
                    [x_coord, y_coord, z_coord],
                    dtype=float,
                )
                direction_vectors_angst.append(unit_vector_angst)

        self.direction_vectors_angst = np.array(direction_vectors_angst, dtype=float)

        print(
            f"Generating grid "
            f"(theta = {self.grid_theta_min_deg:.2f} --> {self.grid_theta_max_deg:.2f}, "
            f"phi = {self.grid_phi_min_deg:.2f} --> {self.grid_phi_max_deg:.2f})."
        )
        print(f"Number of direction vectors generated: {len(self.direction_vectors_angst)}")

        return self.direction_vectors_angst

    def survey_map(
        self,
        unit_vector_angst: np.ndarray,
    ) -> np.ndarray:
        """
        Survey the cryo-EM map along a unit direction vector.

        This method samples from a pre-smoothed cryo-EM map and computes all
        sampling positions for the direction in a vectorized way.

        Parameters
        ----------
        unit_vector_angst
            Unit direction vector.

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with columns:
            [distance_traveled_angst, sampled_intens]
        """
        if self.smoothed_cryo_em_map is None:
            self.prepare_smoothed_map(window_size=1)

        survey_radius_min_pix = int(
            np.floor(self.survey_radius_min_angst / self.angspix)
        )
        survey_radius_max_pix = int(
            np.ceil(self.survey_radius_angst / self.angspix)
        )

        radius_values_pix = np.arange(
            survey_radius_min_pix,
            survey_radius_max_pix + 1,
            dtype=int,
        )
        distance_values_angst = radius_values_pix.astype(float) * self.angspix

        sampling_vectors_angst = (
            self.center_pos_angst[None, :]
            + unit_vector_angst[None, :] * distance_values_angst[:, None]
        )

        sampling_vectors_pix = np.rint(
            sampling_vectors_angst / self.angspix
        ).astype(int)

        sampling_vectors_pix = np.clip(
            sampling_vectors_pix,
            [0, 0, 0],
            self.map_dims_pix - 1,
        )

        sampled_intensities = self.smoothed_cryo_em_map[
            sampling_vectors_pix[:, 0],
            sampling_vectors_pix[:, 1],
            sampling_vectors_pix[:, 2],
        ]

        return np.column_stack((distance_values_angst, sampled_intensities))

    def calculate_raw_density_data(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Survey the map for each stored direction vector.

        If direction vectors have not yet been calculated, this method attempts
        to calculate them from the stored grid limits.

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            Dictionary indexed by direction number. Each item contains:
            - 'unit_vector_angst'
            - 'survey_data'
        """
        if self.direction_vectors_angst is None:
            self.calculate_direction_vectors_angst()

        if self.smoothed_cryo_em_map is None:
            self.prepare_smoothed_map(window_size=1)

        raw_density_data = {}

        for direction_index, unit_vector_angst in enumerate(self.direction_vectors_angst):
            survey_data = self.survey_map(unit_vector_angst)
            raw_density_data[direction_index] = {
                "unit_vector_angst": unit_vector_angst,
                "survey_data": survey_data,
            }

        self.raw_density_data = raw_density_data
        print("Raw density data calculated!")