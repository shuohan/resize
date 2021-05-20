import torch
import torch.nn.functional as F

from .abstract import Resize


def resize(image, dxyz, same_fov=True, target_shape=None, mode='bicubic',
           return_coords=False):
    """Wrapper function to resize an image using numpy.

    See :class:`ResizeNumpy` for more details.

    Args:
        return_coords (bool): Return sampling coordinates if ``True``.

    """
    resizer = ResizeTorch(image, dxyz, same_fov=same_fov,
                          target_shape=target_shape, mode=mode)
    resizer.resize()
    if return_coords:
        return resizer.result, resizer.coords
    else:
        return resizer.result


class ResizeTorch(Resize):
    """Resizes the image with sampling steps dx, dy, and dz in PyTorch.

    Same FOV mode:
        
        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Align first point mode:
        
        |   x   |   x   |   x   |
          | x | x | x | x | x |

    Note:
        * Assume "replication" padding in the same FOV mode.
        * Only support linear interpolation.
        * Only support 2D and 3D images.

    Args:
        image (torch.Tensor): The image to resample with shape B x C x
            spatial_size, the B is the batch size and C is the number of
            channels.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling.
        same_fov (bool): Keep the same FOV if possible when ``True``.
        mode (str): Interpolation mode. See documents for
            :func:`torch.nn.functional.grid_sample`.
        target_shape (tuple[int]): The target spatial shape if not ``None``.

    """
    def __init__(self, image, dxyz, same_fov=True, target_shape=None,
                 mode='bicubic'):
        self.mode = mode
        super().__init__(image, dxyz, same_fov, target_shape)
        self._old_shape = self.image.shape[2:]

    def _check_shape(self):
        super()._check_shape()
        assert len(self.image.shape) == len(self.dxyz) + 2

    @property
    def coords(self):
        """Returns sampling coordinates.

        The shape is B x spatial_size x M coordinate array. This array has the
        same spatial shape with the resuling interpolated image. B is the number
        of samples in this mini-batch of images, but they share the same
        sampling indices. M is the number of cooridnates for each point to
        sample. The spatial size can be 2D or 3D.

        """
        return self._coords

    def _format_coords(self):
        # Map into the coordinates into [-1, 1] as required by F.grid_sample
        self._coords = self._normalize_coords()
        self._coords = torch.meshgrid(*self._coords)
        self._coords = [c[None, ..., None] for c in self._coords]
        # Reverse the order of  coordinates for F.grid_sample
        self._coords.reverse()
        self._coords = torch.cat(self._coords, -1)
        # The first dim of coords is batch size. PyTorch needs it to be the same
        # with the images to sample.
        repeats = [self.image.shape[0]] + [1] * (self._coords.ndim - 1)
        self._coords = self._coords.repeat(repeats)
        self._coords = self._coords.to(self.image)

    def _normalize_coords(self):
        if self.same_fov:
            fov = [stop - start - 1
                   for start, stop in zip(self._old_fov[0], self._old_fov[1])]
        else:
            fov = self._old_fov
        return [torch.tensor(c / f * 2 - 1) for c, f in zip(self._coords, fov)]

    def _resize(self):
        self._result = F.grid_sample(self.image, self._coords, mode=self.mode,
                                     align_corners=True, padding_mode='border')
