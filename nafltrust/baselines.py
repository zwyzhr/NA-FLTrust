import math
import numpy as np
import torch

def classifier_slice(model):
    parameters = list(model.parameters())
    offset = sum((p.numel() for p in parameters[:-2]))
    return slice(offset, offset + parameters[-2].numel())

def classifier_defense(start, updates, model, data_sizes, method, seed, round_index, epsilon=0.01):
    block = classifier_slice(model)
    reference = start[block].double()
    local = (start.double()[None, :] + updates)[:, block]
    if reference.norm() == 0 or (local.norm(dim=1) == 0).any():
        return (torch.zeros_like(start), False)
    reference = reference / reference.norm()
    local = local / local.norm(dim=1, keepdim=True)
    if method == 'dual_defense':
        sigma = math.sqrt(2 * math.log(1.25 * reference.numel())) / epsilon
        noise = np.random.default_rng(100000 * seed + round_index).normal(0, sigma)
        reference = reference + float(noise)
    scores = (local @ reference).abs()
    spread = scores.max() - scores.min()
    selected = torch.ones(len(scores), dtype=torch.bool, device=scores.device) if spread == 0 else (scores - scores.min()) / spread >= ((scores - scores.min()) / spread).mean()
    if not selected.any():
        return (torch.zeros_like(start), False)
    weight = torch.as_tensor(data_sizes, dtype=updates.dtype, device=updates.device) if method == 'dual_defense' else torch.ones_like(scores)
    weight = weight * selected
    weight = weight / weight.sum()
    return ((updates * weight[:, None]).sum(0), True)
