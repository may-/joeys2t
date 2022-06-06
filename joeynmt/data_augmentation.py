# coding: utf-8
"""
Data Augmentation
"""
import math
from typing import Optional
import numpy as np


class SpecAugment:
    """
    SpecAugment (https://arxiv.org/abs/1904.08779)
    cf.) https://github.com/pytorch/fairseq/blob/main/fairseq/data/audio/feature_transforms/specaugment.py
    """
    def __init__(self,
                 freq_mask_n: int = 2,
                 freq_mask_f: int = 27,
                 time_mask_n: int = 2,
                 time_mask_t: int = 40,
                 time_mask_p: float = 1.0,
                 mask_value: Optional[float] = None):

        self.freq_mask_n = freq_mask_n
        self.freq_mask_f = freq_mask_f
        self.time_mask_n = time_mask_n
        self.time_mask_t = time_mask_t
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames, num_freqs = spectrogram.shape
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            return spectrogram

        if num_freqs < self.freq_mask_f:
            return spectrogram

        for _i in range(self.freq_mask_n):
            f = np.random.randint(0, self.freq_mask_f)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0 : f0 + f] = mask_value

        max_time_mask_t = min(
            self.time_mask_t, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_t < 1:
            return distorted

        for _i in range(self.time_mask_n):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        assert distorted.shape == spectrogram.shape
        return distorted

    def __repr__(self):
        return (f"{self.__class__.__name__}(freq_mask_n={self.freq_mask_n}, "
                f"freq_mask_f={self.freq_mask_f}, time_mask_n={self.time_mask_n}, "
                f"time_mask_t={self.time_mask_t}, time_mask_p={self.time_mask_p})")


class CMVN:
    """
    CMVN: Cepstral Mean and Variance Normalization (Utterance-level)
    cf.) https://github.com/pytorch/fairseq/blob/main/fairseq/data/audio/feature_transforms/utterance_cmvn.py
    """

    def __init__(self, norm_means: bool = True, norm_vars: bool = True, before: bool = True):
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.before = before

    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        assert orig_shape == x.shape
        return x

    def __repr__(self):
        return (f"{self.__class__.__name__}(norm_means={self.norm_means}, "
                f"norm_vars={self.norm_vars}, before={self.before})")
