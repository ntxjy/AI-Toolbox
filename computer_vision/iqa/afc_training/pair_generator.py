# -*- coding: utf-8 -*-
"""Pair generation utilities for 2AFC IQA training.

This module provides helper functions to build large amounts of
``(i, j)`` pairs from each image scene and label which image has
higher quality.  The generated pairs can then be used for pair-wise
or list-wise 2AFC training.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, Iterable, List, Sequence, Tuple


# -----------------------------------------------------------------------------
# Pair generation
# -----------------------------------------------------------------------------

def _sample_index_pairs(
    n: int,
    num_pairs: int | None = None,
    rnd: random.Random | None = None,
) -> List[Tuple[int, int]]:
    """Sample index pairs for ``n`` items.

    Parameters
    ----------
    n: int
        Number of items contained in a single scene.
    num_pairs: int, optional
        Maximum number of pairs to generate.  If ``None`` all ``n`` choose
        2 pairs are returned.
    rnd: :class:`random.Random`, optional
        Random number generator for reproducibility.

    Returns
    -------
    list
        A list of index tuples ``(i, j)`` with ``i < j``.
    """

    if n < 2:
        return []

    pairs = list(itertools.combinations(range(n), 2))
    if num_pairs is not None and len(pairs) > num_pairs:
        rnd = rnd or random
        rnd.shuffle(pairs)
        pairs = pairs[:num_pairs]
    return pairs


def generate_pairs(
    items: Sequence[Tuple[str, float]],
    num_pairs: int | None = None,
    rnd: random.Random | None = None,
) -> List[Tuple[str, str, int]]:
    """Generate labelled pairs from a single scene.

    Parameters
    ----------
    items: sequence
        Sequence of ``(path, score)`` pairs for a single scene.
    num_pairs: int, optional
        Maximum number of pairs to sample from this scene.  If ``None``
        all possible pairs are generated.
    rnd: :class:`random.Random`, optional
        Random generator for reproducible sampling.

    Returns
    -------
    list
        A list of tuples ``(img_i, img_j, label)``. ``label`` is ``1`` when
        ``img_i`` has a higher MOS/quality score than ``img_j`` otherwise ``0``.
    """

    index_pairs = _sample_index_pairs(len(items), num_pairs, rnd)
    labelled_pairs: List[Tuple[str, str, int]] = []
    for i, j in index_pairs:
        img_i, score_i = items[i]
        img_j, score_j = items[j]
        label = 1 if score_i > score_j else 0
        labelled_pairs.append((img_i, img_j, label))
    return labelled_pairs


def generate_dataset_pairs(
    scenes: Dict[str, Sequence[Tuple[str, float]]],
    num_pairs_per_scene: int | None = None,
    seed: int | None = None,
) -> List[Tuple[str, str, int]]:
    """Generate labelled pairs for all scenes in a dataset.

    Parameters
    ----------
    scenes: dict
        Mapping from ``scene_id`` to a sequence of ``(path, score)`` tuples.
    num_pairs_per_scene: int, optional
        Maximum number of pairs to sample for each scene.  If ``None``
        all pairs from each scene are used.
    seed: int, optional
        Seed for the internal random number generator.

    Returns
    -------
    list
        Concatenation of pairs generated for all scenes.
    """

    rnd = random.Random(seed)
    all_pairs: List[Tuple[str, str, int]] = []
    for _scene_id, items in scenes.items():
        all_pairs.extend(generate_pairs(items, num_pairs_per_scene, rnd))
    return all_pairs


__all__ = [
    "generate_pairs",
    "generate_dataset_pairs",
]
