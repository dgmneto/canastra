"""Flat parameter genomes ↔ torch modules, and the pinned weights-JSON format."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

FORMAT = "canastra-weights@1"

Arch = dict[str, Any]  # {"obs": int, "act": int, "trunk": list[int], "head": list[int], "activation": "tanh"}


def layer_shapes(arch: Arch) -> list[tuple[str, int, int]]:
    """(name, out, in) for every layer, in genome order."""
    shapes: list[tuple[str, int, int]] = []
    prev = int(arch["obs"])
    for i, width in enumerate(arch["trunk"]):
        shapes.append((f"trunk.{i}", int(width), prev))
        prev = int(width)
    prev += int(arch["act"])
    for i, width in enumerate(arch["head"]):
        shapes.append((f"head.{i}", int(width), prev))
        prev = int(width)
    shapes.append(("head.out", 1, prev))
    return shapes


def genome_size(arch: Arch) -> int:
    return sum(out * inn + out for _, out, inn in layer_shapes(arch))


def random_genome(arch: Arch, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, genome_size(arch)).astype(np.float32)


def to_modules(vec: np.ndarray, arch: Arch) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
    """Split the flat genome into (trunk, head) Linear stacks. Deterministic."""
    trunk = torch.nn.ModuleList()
    head = torch.nn.ModuleList()
    offset = 0
    for name, out, inn in layer_shapes(arch):
        weight = torch.from_numpy(vec[offset : offset + out * inn].reshape(out, inn).copy())
        offset += out * inn
        bias = torch.from_numpy(vec[offset : offset + out].copy())
        offset += out
        layer = torch.nn.Linear(inn, out)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        (trunk if name.startswith("trunk") else head).append(layer)
    assert offset == vec.size
    return trunk, head


def from_modules(trunk: torch.nn.ModuleList, head: torch.nn.ModuleList) -> np.ndarray:
    parts = []
    for layer in [*trunk, *head]:
        parts.append(layer.weight.detach().numpy().ravel())
        parts.append(layer.bias.detach().numpy().ravel())
    return np.concatenate(parts).astype(np.float32)


def save_json(path: str, arch: Arch, vec: np.ndarray) -> None:
    params: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, out, inn in layer_shapes(arch):
        w = vec[offset : offset + out * inn]
        offset += out * inn
        b = vec[offset : offset + out]
        offset += out
        params[f"{name}.weight"] = {"shape": [out, inn], "data": np.round(w, 6).tolist()}
        params[f"{name}.bias"] = {"shape": [out], "data": np.round(b, 6).tolist()}
    with open(path, "w") as handle:
        json.dump({"format": FORMAT, "arch": arch, "params": params}, handle)


def load_json(path: str) -> tuple[Arch, np.ndarray]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported weights format: {payload.get('format')!r}")
    arch = payload["arch"]
    if arch.get("activation") != "tanh":
        raise ValueError("only tanh weights are supported")
    vec = np.zeros(genome_size(arch), dtype=np.float32)
    offset = 0
    for name, out, inn in layer_shapes(arch):
        for key, size in ((f"{name}.weight", out * inn), (f"{name}.bias", out)):
            entry = payload["params"][key]
            if entry["shape"] != ([out, inn] if "weight" in key else [out]):
                raise ValueError(f"{key}: shape {entry['shape']} does not match the arch")
            if len(entry["data"]) != size:
                raise ValueError(f"{key}: {len(entry['data'])} values, expected {size}")
            vec[offset : offset + size] = np.asarray(entry["data"], dtype=np.float32)
            offset += size
    return arch, vec