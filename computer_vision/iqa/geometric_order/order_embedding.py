"""Utility to generate geometric order embeddings for image quality lists.

This module produces simple geometric order embeddings given lists of
reference and distorted image quality scores.  The embeddings capture
relative relationships such as differences and ratios, which are useful for
list-wise ranking approaches in image quality assessment.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def geometric_order_embedding(
    reference: Iterable[float],
    distorted: Iterable[float],
    eps: float = 1e-8,
) -> np.ndarray:
    """Generate geometric order embeddings.

    Parameters
    ----------
    reference: Iterable[float]
        Quality scores for the pristine reference images.
    distorted: Iterable[float]
        Quality scores for the distorted images.  Must have the same length
        as ``reference``.
    eps: float, optional
        Small value used to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(N, 3)`` where ``N`` is the number of image
        pairs.  For each pair the embedding contains three components:
        ``difference`` (distorted - reference), ``ratio`` (distorted / reference)
        and ``log_ratio`` (log(distorted) - log(reference)).

    Notes
    -----
    The embedding is a compact representation of the geometric order between
    two lists and can be fed into learning-to-rank models.
    """
    ref = np.asarray(list(reference), dtype=np.float32)
    dis = np.asarray(list(distorted), dtype=np.float32)
    if ref.shape != dis.shape:
        raise ValueError("`reference` and `distorted` must have the same shape")
    # Difference captures additive deviations
    difference = dis - ref
    # Ratio captures multiplicative deviations
    ratio = dis / (ref + eps)
    # Log-ratio provides a symmetric measure robust to scale
    log_ratio = np.log(dis + eps) - np.log(ref + eps)
    embedding = np.stack([difference, ratio, log_ratio], axis=-1)
    return embedding
