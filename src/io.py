import numpy as np
import mrcfile


def load_mrc_map(mrc_file_path: str) -> np.ndarray:
    """
    Load a cryo-EM map from an .mrc file.

    Parameters
    ----------
    mrc_file_path
        Path to the input .mrc file.

    Returns
    -------
    np.ndarray
        3D cryo-EM density map.
    """
    with mrcfile.open(mrc_file_path, permissive=True) as mrc:
        cryo_em_map = np.array(mrc.data)

    return cryo_em_map