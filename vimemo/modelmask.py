from __future__ import annotations

import numpy as np
import pandas as pd


class ModelMask:
    """
    Convert a nonzero-valued model mask array into a coordinate dataframe.

    The spherical-coordinate convention matches CryoEMSurvey:
    - theta: azimuthal angle in the xy-plane from arctan2(y, x)
    - phi: polar angle from the positive z-axis
    """

    def __init__(
        self,
        cryo_em_map: np.ndarray,
        pixel_size: float,
        *,
        keep_value: bool = False,
        value_col: str = "density",
    ) -> None:
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive.")

        self.cryo_em_map = np.asarray(cryo_em_map)
        if self.cryo_em_map.ndim != 3:
            raise ValueError("cryo_em_map must be a 3D NumPy array.")

        self.pixel_size = float(pixel_size)
        self.keep_value = bool(keep_value)
        self.value_col = str(value_col)

        self.voxel_dataframe = self._build_dataframe()
        self.cartesian_limit_values = self._calculate_cartesian_limits()
        self.spherical_limit_values = self._calculate_spherical_limits()

    def _build_dataframe(self) -> pd.DataFrame:
        nonzero_i, nonzero_j, nonzero_k = np.nonzero(self.cryo_em_map)

        nz_dim, ny_dim, nx_dim = self.cryo_em_map.shape
        center_x = (nx_dim - 1) / 2.0
        center_y = (ny_dim - 1) / 2.0
        center_z = (nz_dim - 1) / 2.0

        x_coord = (nonzero_i - center_x) * self.pixel_size
        y_coord = (nonzero_j - center_y) * self.pixel_size
        z_coord = (nonzero_k - center_z) * self.pixel_size

        rho = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
        theta = np.arctan2(y_coord, x_coord)
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.arccos(np.where(rho == 0, np.nan, z_coord / rho))

        data: dict[str, np.ndarray] = {
            "x_coord": x_coord,
            "y_coord": y_coord,
            "z_coord": z_coord,
            "rho": rho,
            "theta": theta,
            "phi": phi,
        }

        if self.keep_value:
            data[self.value_col] = self.cryo_em_map[nonzero_i, nonzero_j, nonzero_k]

        return pd.DataFrame(data)

    def to_dataframe(self) -> pd.DataFrame:
        return self.voxel_dataframe.copy()

    def _calculate_cartesian_limits(self) -> dict[str, float]:
        if self.voxel_dataframe.empty:
            raise ValueError("The model mask dataframe is empty.")

        return {
            "x_min": float(self.voxel_dataframe["x_coord"].min()),
            "x_max": float(self.voxel_dataframe["x_coord"].max()),
            "y_min": float(self.voxel_dataframe["y_coord"].min()),
            "y_max": float(self.voxel_dataframe["y_coord"].max()),
            "z_min": float(self.voxel_dataframe["z_coord"].min()),
            "z_max": float(self.voxel_dataframe["z_coord"].max()),
        }

    def _calculate_spherical_limits(self) -> dict[str, float]:
        if self.voxel_dataframe.empty:
            raise ValueError("The model mask dataframe is empty.")

        return {
            "rho_min": float(self.voxel_dataframe["rho"].min()),
            "rho_max": float(self.voxel_dataframe["rho"].max()),
            "theta_min_deg": float(np.degrees(self.voxel_dataframe["theta"].min())),
            "theta_max_deg": float(np.degrees(self.voxel_dataframe["theta"].max())),
            "phi_min_deg": float(np.degrees(self.voxel_dataframe["phi"].min())),
            "phi_max_deg": float(np.degrees(self.voxel_dataframe["phi"].max())),
        }

    def cartesian_limits(self) -> dict[str, float]:
        return dict(self.cartesian_limit_values)

    def spherical_limits(self) -> dict[str, float]:
        return dict(self.spherical_limit_values)

    def print_cartesian_limits(self) -> None:
        limits = self.cartesian_limits()
        print("Cartesian limits (A):")
        print(f"x: {limits['x_min']:.3f} to {limits['x_max']:.3f}")
        print(f"y: {limits['y_min']:.3f} to {limits['y_max']:.3f}")
        print(f"z: {limits['z_min']:.3f} to {limits['z_max']:.3f}")

    def print_spherical_limits(self) -> None:
        limits = self.spherical_limits()
        print("Spherical limits:")
        print(f"rho: {limits['rho_min']:.3f} to {limits['rho_max']:.3f} A")
        print(
            f"theta: {limits['theta_min_deg']:.3f} to "
            f"{limits['theta_max_deg']:.3f} deg"
        )
        print(
            f"phi: {limits['phi_min_deg']:.3f} to "
            f"{limits['phi_max_deg']:.3f} deg"
        )
