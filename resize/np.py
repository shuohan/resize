"""Resize with correct sampling step implemented with numpy

"""
import numpy as np
from scipy.ndimage import map_coordinates

from .utils import _calc_old_fov_same_fov, _calc_old_fov_align_first
from .utils import _calc_new_shape_same_fov, _calc_new_shape_align_first
from .utils import _calc_new_fov_same_fov, _calc_new_fov_align_first


def resize(image, dxyz, order=3, same_fov=True, return_coords=False):
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
        return_coords (bool): Return sampling coordinates if ``True``.

    Returns
    -------
    result : numpy.ndarray
        The resampled image.
    coords : numpy.ndarray (optional)
        The sampling coordiantes. See :func:`calc_sampling_coords`.

    """
    coords, new_shape = calc_sampling_coords(image.shape, dxyz, same_fov)
    result = map_coordinates(image, coords, mode='nearest', order=order)
    result = result.reshape(new_shape)
    if return_coords:
        return result, coords
    else:
        return result


def calc_sampling_coords(shape, dxyz, same_fov=True):
    """Calculates sampling coordinates.

    Args:
        shape (tuple[int]): The shape of the image to resize.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling.
        same_fov (bool): Keep the same FOV if possible when ``True``.

    Returns
    -------
    coords : numpy.ndarray
        The M x N indices array. N is the number of points to sample. M is the
        point dimension (e.g., 1, 2, 3). Each column is the coordinates vector
        of a point.
    new_shape : (tuple[int])
        The shape of the resulting image. ``prod(new_shape)`` is equal to N, the
        number of points in ``coords``.

    """
    if same_fov:
        old_fov = _calc_old_fov_same_fov(shape)
        new_shape = _calc_new_shape_same_fov(shape, dxyz)
        new_fov = _calc_new_fov_same_fov(old_fov, new_shape, dxyz)
        coords = _calc_sampling_coords_same_fov(new_fov, dxyz)
    else:
        old_fov = _calc_old_fov_align_first(shape)
        new_shape = _calc_new_shape_align_first(shape, dxyz)
        new_fov = _calc_new_fov_align_first(new_shape, dxyz)
        coords = _calc_sampling_coords_align_first(new_fov, dxyz)
    return coords, new_shape


def _calc_sampling_coords_same_fov(new_fov, dxyz):
    indices = [np.arange(l + d/2, r - d/4, d)
               for l, r, d in zip(new_fov[0], new_fov[1], dxyz)]
    grid = np.meshgrid(*indices, indexing='ij')
    grid = np.array([g.flatten() for g in grid])
    return grid


def _calc_sampling_coords_align_first(new_fov, dxyz):
    indices = [np.arange(0, f + d/4, d) for f, d in zip(new_fov, dxyz)]
    grid = np.meshgrid(*indices, indexing='ij')
    grid = [g.flatten() for g in grid]
    return np.array(grid)
