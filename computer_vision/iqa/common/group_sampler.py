import random
from collections import defaultdict
from typing import List, Sequence

from torch.utils.data import Sampler


class GroupSampler(Sampler[int]):
    """Group-balanced sampler with cross-scene parameter alignment.

    Each sample in the dataset should have an associated ``scene`` and
    ``param`` identifier.  The sampler groups samples by ``param`` and ensures
    that indices from different scenes but with the same parameter value are
    aligned.  During iteration it yields indices so that every parameter group
    contributes the same number of samples from every scene.

    Args:
        scenes: sequence of scene identifiers for each sample in the dataset.
        params: sequence of parameter identifiers for each sample.
        shuffle: if ``True`` the aligned groups are shuffled before yielding.
    """

    def __init__(self, scenes: Sequence[int], params: Sequence[int], shuffle: bool = True) -> None:
        assert len(scenes) == len(params), "scenes and params must be the same length"
        self.scenes = list(scenes)
        self.params = list(params)
        self.shuffle = shuffle

        # Build mapping: param -> scene -> [indices]
        group_dict: dict = defaultdict(lambda: defaultdict(list))
        for idx, (s, p) in enumerate(zip(self.scenes, self.params)):
            group_dict[p][s].append(idx)

        self.scene_ids: List[int] = sorted(set(self.scenes))

        # Ensure cross-scene alignment by constructing packets of indices
        param_packets = {}
        for p, scene_map in group_dict.items():
            # Skip parameter groups that do not contain all scenes
            if any(s not in scene_map for s in self.scene_ids):
                continue
            # Sort indices in each scene for deterministic behaviour
            for indices in scene_map.values():
                indices.sort()
            min_len = min(len(scene_map[s]) for s in self.scene_ids)
            packets = [[scene_map[s][i] for s in self.scene_ids] for i in range(min_len)]
            param_packets[p] = packets

        # Balance the number of packets across all parameters
        if param_packets:
            min_packets = min(len(v) for v in param_packets.values())
        else:
            min_packets = 0
        packets: List[List[int]] = []
        for p, p_packets in param_packets.items():
            packets.extend(p_packets[:min_packets])

        self.packets = packets

    def __iter__(self):
        packets = list(self.packets)
        if self.shuffle:
            random.shuffle(packets)
        for packet in packets:
            for idx in packet:
                yield idx

    def __len__(self) -> int:  # number of indices this sampler will provide
        return len(self.packets) * len(self.scene_ids)
