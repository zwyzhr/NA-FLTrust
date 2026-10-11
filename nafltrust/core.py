from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class Config:
    degree: int = 4
    beta: float = 0.05
    kappa: float = 1.0
    eta: float = 1.0
    denominator_floor: float = 1e-06
    rounding_bits: int = 24
    unit_diagnostic: bool = False

    def __post_init__(self):
        if not (self.unit_diagnostic and self.degree == 1) and (self.degree < 2 or self.degree % 2):
            raise ValueError('degree must be positive and even, except explicit unit-input k=1 diagnostic')
        if not 0 < self.beta <= 1 or not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError('invalid safeguard')
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError('invalid eta')

class RoundSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bits):
        return torch.round(x * 2 ** bits) / 2 ** bits

    @staticmethod
    def backward(ctx, grad):
        return (grad, None)

class Clip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x)
        ctx.lo = lo
        ctx.hi = hi
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return (grad * ((x > ctx.lo) & (x < ctx.hi)), None, None)

def normalize_rows(x):
    norms = x.norm(dim=-1, keepdim=True)
    return x / torch.where(norms > 0, norms, torch.ones_like(norms))

def safeguard(v, g, cfg=Config()):
    a = v @ g
    r = v - a * g
    ah = Clip.apply(a, cfg.beta, 1.0)
    scale = Clip.apply(cfg.kappa * ah / r.norm().clamp_min(1e-30), 0.0, 1.0)
    return ah * g + scale * r

def release(numerator, denominator, root, cfg=Config(), surrogate=False):
    radius = root.norm()
    zero = torch.zeros_like(root)
    if radius == 0 or not torch.isfinite(radius) or (not torch.isfinite(root).all()):
        return (zero, False)
    quantize = lambda x: RoundSTE.apply(x, cfg.rounding_bits) if surrogate else torch.round(x * 2 ** cfg.rounding_bits) / 2 ** cfg.rounding_bits
    num, den = (quantize(numerator), quantize(denominator))
    if not torch.isfinite(num).all() or not torch.isfinite(den) or den <= cfg.denominator_floor:
        return (zero, False)
    return (cfg.eta * radius * safeguard(num / den, root / radius, cfg), True)

def encrypted_inputs_reference(updates, root, cfg=Config()):
    if cfg.unit_diagnostic and bool((updates.norm(dim=1) > 1 + 1e-08).any()):
        raise ValueError('unit-input diagnostic rejects amplification')
    g = root / root.norm()
    scores = updates @ g
    weights = ((scores + 1) / 2) ** cfg.degree
    norms2 = updates.square().sum(1)
    return ((weights[:, None] * updates).sum(0), (weights * norms2).sum(), weights, norms2)

def aggregate(updates, root, method='na', cfg=Config(), byzantine=0, surrogate=False):
    if updates.ndim != 2 or updates.shape[1:] != root.shape or len(updates) == 0:
        raise ValueError('expected n x d updates and d root')
    if not torch.isfinite(updates).all() or not torch.isfinite(root).all():
        return (torch.zeros_like(root), False)
    if method in ('fedavg', 'average'):
        return (updates.mean(0), True)
    if method == 'median':
        return (updates.quantile(0.5, dim=0), True)
    if method == 'trimmed_mean':
        if byzantine < 0 or 2 * byzantine >= len(updates):
            raise ValueError('trimmed mean requires n > 2b')
        z = updates.sort(dim=0).values
        return (z[byzantine:len(z) - byzantine].mean(0), True)
    if method == 'krum':
        if byzantine < 0 or len(updates) < 2 * byzantine + 3:
            raise ValueError('Krum requires n >= 2b+3')
        d = torch.cdist(updates, updates).square()
        d.fill_diagonal_(float('inf'))
        scores = d.sort(dim=1).values[:, :len(updates) - byzantine - 2].sum(1)
        return (updates[scores.argmin()], True)
    if root.norm() == 0 or not torch.isfinite(root.norm()):
        return (torch.zeros_like(root), False)
    g = root / root.norm()
    if method == 'fltrust':
        u = normalize_rows(updates)
        weights = Clip.apply(u @ g, 0.0, float('inf'))
        den = weights.sum()
        return (cfg.eta * root.norm() * (weights[:, None] * u).sum(0) / den, True) if den > cfg.denominator_floor else (root * 0, False)
    num, den, weights, _ = encrypted_inputs_reference(updates, root, cfg)
    if method == 'na':
        return release(num, den, root, cfg, surrogate)
    if method not in ('polynomial_safeguard', 'polynomial_clip'):
        raise ValueError(method)
    quantize = lambda x: RoundSTE.apply(x, cfg.rounding_bits) if surrogate else torch.round(x * 2 ** cfg.rounding_bits) / 2 ** cfg.rounding_bits
    num, den = (quantize(num), quantize(weights.sum()))
    if not torch.isfinite(num).all() or not torch.isfinite(den) or den <= cfg.denominator_floor:
        return (torch.zeros_like(root), False)
    v = num / den
    if method == 'polynomial_safeguard':
        v = safeguard(v, g, cfg)
    else:
        v = v / v.norm().clamp_min(1.0)
    return (cfg.eta * root.norm() * v, True)
