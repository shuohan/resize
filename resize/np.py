"""Resize with correct sampling step implemented with numpy

"""
import numpy as np
from scipy.ndimage import map_coordinates

from .abstract import Resize


def resize(image, dxyz, same_fov=True, target_shape=None, coords=None, order=3,
           return_coords=False):
    """Wrapper function to resize an image using numpy.

    See :class:`ResizeNumpy` for more details.

    Args:
        return_coords (bool): Return sampling coordinates if ``True``.

    """
    resizer = ResizeNumpy(image, dxyz, same_fov=same_fov,
                          target_shape=target_shape, order=order)
    resizer.resize()
    if return_coords:
        return resizer.result, resizer.coords
    else:
        return resizer.result


class ResizeNumpy(Resize):
    """Resizes the image with sampling steps dx, dy, and dz.

    Same FOV mode:
        
        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Align first point mode:
        
        |   x   |   x   |   x   |
          | x | x | x | x | x |

    Note:
        Assume "replication" padding in the same FOV mode.

    Args:
        image (numpy.ndarray): The image to resample.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling.
        same_fov (bool): Keep the same FOV if possible when ``True``.
        target_shape (tuple[int]): The target spatial shape if not ``None``.
        order (int): B-spline interpolation order.

    """
    def __init__(self, image, dxyz, same_fov=True, target_shape=None,
                 coords=None, order=3):
        self.order = order
        super().__init__(image, dxyz, same_fov, target_shape, coords)

    def _check_shape(self):
        super()._check_shape()
        assert len(self.image.shape) == len(self.dxyz)

    @property
    def coords(self):
        """Returns the sampling coordinates.

        The shape is M x N sampling coordinate array. N is the number of points
        to sample. It is equal to product of the shape of :meth:`result`.
        M is the point dimension (e.g., 1, 2, 3).  Each column is the
        coordinates vector of a point.

        """
        return self._coords

    def _format_coords(self):
        self._coords = np.meshgrid(*self._coords, indexing='ij')
        self._coords = np.array([c.flatten() for c in self._coords])

    def _resize(self):
        self._result = map_coordinates(self.image, self.coords,
                                       mode='nearest', order=self.order)
        self._result = self._result.reshape(self._new_shape)
