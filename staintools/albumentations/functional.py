import numpy
from .. import StainAugmentor, LuminosityStandardizer
from ..utils.exceptions import TissueMaskException


def random_staining(
    img,
    method: str,
    sigma1: float = 0.2,
    sigma2: float = 0.2,
    augment_background: bool = True,
):
    dtype = img.dtype
    augmentor = StainAugmentor(
        method=method,
        sigma1=sigma1,
        sigma2=sigma2,
        augment_background=augment_background,
        is_parallel=False,
    )
    augimg = LuminosityStandardizer.standardize(img)
    try:
        augmentor.fit(augimg)
    except TissueMaskException:
        return img
    augimg = augmentor.pop()
    augimg = augimg.astype(dtype)
    return augimg
