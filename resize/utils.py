def _calc_old_fov_align_first(old_shape):
    """Assumes the FOV starts from the first and ends at the last point."""
    step_size = 1
    fov = tuple((s - 1) * step_size for s in old_shape)
    return fov


def _calc_new_fov_align_first(new_shape, dxyz):
    """The new FOV also starts from the first point."""
    fov = tuple((s - 1) * d for s, d in zip(new_shape, dxyz))
    return fov


def _calc_new_shape_align_first(old_shape, dxyz):
    """The largest number of points to make sure new FOV is within the old."""
    return tuple(np.floor((s - 1) / d) + 1 for s, d in zip(old_shape, dxyz))


def _calc_old_fov_same_fov(old_shape):
    """Calculates the FOV of the original image.

    Suppose the left boundaries are at (-0.5, -0.5, -0.5), then the first voxel
    is at (0, 0, 0). Assume the step size is 1.

    """
    step_size = 1
    lefts = (-0.5, ) * len(old_shape)
    rights = tuple(l + s * step_size for l, s in zip(lefts, old_shape))
    return lefts, rights


def _calc_new_fov_same_fov(old_fov, new_shape, dxyz):
    """Calculates the FOV of the resulting image.

    Assume the old and new FOV have the same center, then the new FOV is shifted
    from the old FOV by half of the size difference.

    """
    old_size = [r - l for l, r in zip(*old_fov)]
    new_size = [s * d for s, d in zip(new_shape, dxyz)]
    size_diff = [(o - n) / 2 for o, n in zip(old_size, new_size)]
    lefts = tuple(l + sd for l, sd in zip(old_fov[0], size_diff))
    rights = tuple(r - sd for r, sd in zip(old_fov[1], size_diff))
    return lefts, rights


def _calc_new_shape(old_shape, dxyz):
    """Calculates the shape of the interpolated image."""
    return tuple(round(s / d) for s, d in zip(old_shape, dxyz))
