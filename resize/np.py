"""Resize with correct sampling step implemented with numpy

"""
import numpy as np
from scipy.ndimage import map_coordinates

from .utils import _calc_old_fov_same_fov, _calc_old_fov_align_first
from .utils import _calc_new_shape_same_fov, _calc_new_shape_align_first
from .utils import _calc_new_fov_same_fov, _calc_new_fov_align_first


def resize(image, dxyz, order=3, same_fov=True):
    """Resize the image with sampling steps dx, dy, and dz.

    Same FOV mode:
        
        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Align first point mode:
        
        |   x   |   x   |   x   |
          | x | x | x | x | x |

    Note:
        Assum "replication" padding in the same FOV mode.

    Args:
        image (numpy.ndarray): The image to resample.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling.
        order (int): B-spline interpolation order.
        same_fov (bool): Keep the same FOV if possible when ``True``.

    Returns:
        numpy.ndarray: The resampled image.

    """
    if same_fov:
        old_fov = _calc_old_fov_same_fov(image.shape)
        new_shape = _calc_new_shape_same_fov(image.shape, dxyz)
        new_fov = _calc_new_fov_same_fov(old_fov, new_shape, dxyz)
        indices = _calc_sampling_indices_same_fov(new_fov, dxyz)
    else:
        old_fov = _calc_old_fov_align_first(image.shape)
        new_shape = _calc_new_shape_align_first(image.shape, dxyz)
        new_fov = _calc_new_fov_align_first(new_shape, dxyz)
        indices = _calc_sampling_indices_align_first(new_fov, dxyz)
    result = map_coordinates(image, indices, mode='nearest', order=order)
    result = result.reshape(new_shape)
    return result


def _calc_sampling_indices_same_fov(new_fov, dxyz):
    indices = [np.arange(l + d/2, r - d/4, d)
               for l, r, d in zip(new_fov[0], new_fov[1], dxyz)]
    grid = np.meshgrid(*indices, indexing='ij')
    grid = [g.flatten() for g in grid]
    return np.array(grid)


def _calc_sampling_indices_align_first(new_fov, dxyz):
    indices = [torch.arange(0, f + d/4, d) for f, d in zip(new_fov, dxyz)]
    grid = np.meshgrid(*indices, indexing='ij')
    grid = [g.flatten() for g in grid]
    return np.array(grid)
