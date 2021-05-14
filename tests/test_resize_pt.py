#!/usr/bin/env python

import torch
from resize.pt import resize


def test_pt():
    # Same FOV
    x1 = torch.arange(5).float()[None, None, :, None]
    x2 = torch.arange(7).float()[None, None, None, :]
    x = (x1 + x2)
    y = resize(x, (0.7, 2.4))
    assert y.shape == (1, 1, 7, 3)
    assert torch.allclose(y[0, 0, :, 0],
                          torch.tensor((0.6, 1.2, 1.9, 2.6, 3.3, 4, 4.6)))
    assert torch.allclose(y[0, 0, 0, :],
                          torch.tensor((0.6, 3, 5.4)))

    x1 = torch.arange(6)[:, None, None]
    x2 = torch.arange(7)[None, :, None]
    x3 = torch.arange(5)[None, None, :]
    x = (x1 + x2 + x3).float()[None, None, ...]
    dxyz = (1.5, 2, 0.3)
    y, coords = resize(x, dxyz, return_coords=True)
    assert y.shape == (1, 1, 4, 4, 17)
    assert torch.allclose(y[0, 0, :, 0, 0],
                          torch.tensor([0.25, 1.75, 3.25, 4.75]).float())
    assert torch.allclose(y[0, 0, 0, :, 0],
                          torch.tensor([0.25, 2.25, 4.25, 6.25]).float())
    assert torch.allclose(y[0, 0, 0, 0, :],
                          torch.tensor([0.25, 0.25, 0.45, 0.75, 1.05, 1.35,
                                        1.65, 1.95, 2.25, 2.55, 2.85, 3.15,
                                        3.45, 3.75, 4.05, 4.25, 4.25]).float())

    # Align first

    y = resize(x, dxyz, same_fov=False)
    assert y.shape == (1, 1, 4, 4, 14)
    assert torch.allclose(y[0, 0, :, 0, 0], torch.tensor([0, 1.5, 3, 4.5]))
    assert torch.allclose(y[0, 0, 0, :, 0], torch.tensor([0, 2, 4, 6]).float())
    assert torch.allclose(y[0, 0, 0, 0, :],
                          torch.tensor([0,   0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1,
                                        2.4, 2.7, 3.0, 3.3, 3.6, 3.9]))

    x1 = torch.arange(5).float()[None, None, :, None]
    x2 = torch.arange(7).float()[None, None, None, :]
    x = (x1 + x2)
    y = resize(x, (0.7, 2.4), same_fov=False)
    assert torch.allclose(y[0, 0, :, 0],
                          torch.tensor([0, 0.7, 1.4, 2.1, 2.8, 3.5]))
    assert torch.allclose(y[0, 0, 0, :],
                          torch.tensor([0, 2.4, 4.8]))

    print('successful')


if __name__ == '__main__':
    test_pt()
