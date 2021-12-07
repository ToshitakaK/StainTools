from albumentations.core.transforms_interface import ImageOnlyTransform
from . import functional as F


class RandomStaining(ImageOnlyTransform):
    def __init__(
        self,
        method: str,
        sigma1: float = 0.2,
        sigma2: float = 0.2,
        augment_background: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.method = method
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background

    def apply(self, img, **params):
        return F.random_staining(
            img, self.method, self.sigma1, self.sigma2, self.augment_background
        )

    def get_transform_init_args_names(self):
        return ("method", "sigma1", "sigma2", "augment_background")
