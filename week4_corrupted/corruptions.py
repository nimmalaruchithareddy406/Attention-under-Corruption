import numpy as np
import cv2
from skimage.filters import gaussian
from skimage import color as sk_color
from scipy.ndimage import map_coordinates

def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x = np.array(x) / 255.
    # Disk kernel logic
    radius, alias_blur = c[0], c[1]
    L = np.arange(-8, 8 + 1) if radius <= 8 else np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    kernel = np.array((X**2 + Y**2) <= radius**2, dtype=np.float32)
    kernel /= np.sum(kernel)
    kernel = cv2.GaussianBlur(kernel, (3,3) if radius <= 8 else (5,5), sigmaX=alias_blur)
    
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    return np.clip(np.array(channels).transpose((1, 2, 0)), 0, 1) * 255

def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]
    x = np.array(x) / 255.
    x = sk_color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk_color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def elastic_transform(image, severity=1):
    IMSIZE = 32
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
         (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
         (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
         (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # Affine transformation used in the Week 3 reference
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size,
         center_square[1] - square_size],
        center_square - square_size
    ])
    pts2 = pts1 + np.random.uniform(
        -c[2], c[2], size=pts1.shape
    ).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1],
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Elastic displacement field
    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]),
            c[1], mode="reflect", truncate=3
        ) * c[0]
    ).astype(np.float32)

    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]),
            c[1], mode="reflect", truncate=3
        ) * c[0]
    ).astype(np.float32)

    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0]),
        np.arange(shape[2])
    )

    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1))
    )

    return np.clip(
        map_coordinates(
            image, indices, order=1, mode="reflect"
        ).reshape(shape),
        0, 1
    ) * 255

class CIFAR10CCorruption(object):
    """PyTorch Transform that applies one of 4 corruptions at a random severity."""
    def __init__(self, p=1.0):
        self.p = p
        self.corruptions = [brightness, contrast, defocus_blur, elastic_transform]

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        
        c_fn = np.random.choice(self.corruptions)
        severity = np.random.randint(1, 6)
        # Convert PIL to NP, corrupt, then back to NP for ToTensor
        img_np = np.array(img)
        corrupted = c_fn(img_np, severity=severity).astype(np.uint8)
        return corrupted
