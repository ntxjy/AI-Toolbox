"""Backbone registry for IQA models."""

from typing import Dict, Any

from .fast_itpn import FastITPN
from .vmamba import VMamba

BACKBONES = {
    "fast_itpn": FastITPN,
    "vmamba": VMamba,
}

def build_backbone(cfg: Dict[str, Any]):
    """Build backbone from configuration.

    Args:
        cfg: Configuration dictionary. Expected keys:
            - name: Backbone name, e.g. ``fast_itpn`` or ``vmamba``.
            - pretrained: Optional path to pretrained weights.
            - freeze: Whether to freeze backbone parameters.
            - kwargs: Optional dictionary passed to backbone constructor.
    Returns:
        Instantiated backbone module.
    """
    name = cfg.get("name")
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}")
    params = cfg.get("kwargs", {})
    params.update({
        "pretrained": cfg.get("pretrained"),
        "freeze": cfg.get("freeze", False),
    })
    backbone_cls = BACKBONES[name]
    return backbone_cls(**params)
