# Resize with Correct Sampling Step

## Sampling step when resizing an image

When resizing a digital image with interpolation, we usually need to sample values at non-integer coordinates. The sampling step, i.e., the separation between two adjacent pixels/voxels to be sampled, then determines the digital resolution in the resulting image. When given a target digital resolution, we usually want our image resizing routines respect the corresponding sampling step. For example, when visualizing a zoomed-in medical image, if the digital resolution calculated from the image header mismatches the sampling step, we will see either a squeezed or stretched image.

There are many Python libraries available to resize an image. However, functions such as `scipy.ndimage.zoom` and `skimage.transform.rescale` do not respect the sampling step. Here we use a very simple experiment to demonstrate this.

```python
import numpy as np
from scipy.ndimage import zoom

sampling_step = 0.7
x = np.arange(6).astype(float)

# Perform a linear interpolation with replication padding
y = zoom(x, 1 / sampling_step, order=1, mode='nearest')
```

With Python version 3.7.6 and Scipy version 1.6.3, we will have y equal to

```python
[0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5.]
```

Since our sampling step is chosen as 0.7, we would expect the resulting array to have approximately 0.7 different between two nearby values. However, we get 0.625 in this case. For scikit-image,

```python
import numpy as np
from skimage.transform import rescale

sampling_step = 0.7
x = np.arange(6).astype(float)

# Perform a linear interpolation with replication padding
y = rescale(x, 1 / sampling_step, order=1, mode='edge')
```

With Python version 3.7.6 and sciki-image version 1.18.1, we have y

```python
[0, 0.5, 1.16666667, 1.83333333, 2.5, 3.16666667, 3.83333333, 4.5, 5]
```

Here the nearby difference is 0.6666667 (ignore the first and last since they are affected by padding). It is not 0.7 either.

What happens here is that since the resulting image should have an integer number of values, these functions choose to change the sampling step (or the scale, the inverse of the sampling) to accommodate that.

However, if we use the PyTorch function `torch.nn.functional.interpolate`:

```python
import numpy as np
import torch
from torch.nn.functional import interpolate

sampling_step = 0.7
x = torch.arange(6).float()
x = x[None, None, ...] # add batch and channel dim

# Perform a linear interpolation with replication padding
y = interpolate(x, scale_factor=1/sampling_step, mode='linear')
```

With Python version 3.7.6 and PyTorch version 1.8.1, we have

```python
[0.0000, 0.5500, 1.2500, 1.9500, 2.6500, 3.3500, 4.0500, 4.7500]
```

Here we have our 0.7 sampling step back (ignore the values that are affected by padding).

## The shift of the resized image

Even if `torch.nn.functional.interpolate` preserves the sampling step
