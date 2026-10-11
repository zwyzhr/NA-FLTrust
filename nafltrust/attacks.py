import math
import torch
from .core import Config, aggregate, normalize_rows

def prepare(raw, b, method, attack='ipm', epsilon=0.01, scale=10.0):
    if b < 0 or b >= len(raw):
        raise ValueError('need at least one benign client')
    z = raw.clone()
    if b and attack in ('ipm', 'scale', 'alie'):
        benign = raw[:-b]
        malicious = benign.mean(0) + -1.5 * benign.std(0, unbiased=False) if attack == 'alie' else -epsilon * benign.mean(0)
        z[-b:] = malicious
    if method in ('na', 'fltrust', 'polynomial_safeguard', 'polynomial_clip'):
        z = normalize_rows(z)
    if b and attack == 'scale':
        z[-b:] = scale * normalize_rows(z[-b:])
    return z

def joint_attack(benign, root, b, objective, method='na', cfg=Config(), seed=11, round_index=51, steps=200, restarts=10):
    if b < 1 or restarts < 4 or steps < 0:
        raise ValueError('joint attack requires b>=1, restarts>=4, steps>=0')
    gen = torch.Generator(device=root.device).manual_seed(100000 * seed + round_index)
    g = root / root.norm()
    starts = [(-g, math.log(c)) for c in (2.0, 4.0, 16.0)] + [(g, 0.0)]
    for _ in range(restarts - 4):
        starts.append((normalize_rows(torch.randn(root.shape, device=root.device, dtype=root.dtype, generator=gen)), float(torch.rand((), generator=gen, device=root.device)) * math.log(16)))
    best, best_loss = (None, -float('inf'))
    for direction, logc in starts:
        u = direction.detach().clone().requires_grad_()
        v = torch.tensor(logc, dtype=root.dtype, device=root.device, requires_grad=True)
        for step in range(steps + 1):
            x = v.exp() * u
            z = torch.cat([benign.detach(), x.expand(b, -1)])
            with torch.no_grad():
                loss = float(objective(aggregate(z, root, method, cfg)[0]))
            if math.isfinite(loss) and loss > best_loss:
                best, best_loss = (x.detach().clone(), loss)
            if step == steps:
                break
            loss = objective(aggregate(z, root, method, cfg, surrogate=True)[0])
            gu, gv = torch.autograd.grad(loss, (u, v))
            with torch.no_grad():
                proposal = u + 0.05 * gu
                if proposal.norm() > 0:
                    u.copy_(normalize_rows(proposal))
                v.add_(0.1 * gv).clamp_(0.0, math.log(16))
    if best is None:
        raise RuntimeError('no finite attack candidate')
    return (best, best_loss)

def project_trust_cap(direction, reference, minimum_alignment):
    if direction.ndim != 1 or direction.shape != reference.shape or direction.numel() < 2:
        raise ValueError('direction and reference must be matching vectors of dimension >=2')
    if not -1 <= minimum_alignment <= 1:
        raise ValueError('alignment must lie in [-1,1]')
    if not torch.isfinite(direction).all() or not torch.isfinite(reference).all() or reference.norm() <= 0:
        raise ValueError('finite vectors and nonzero reference required')
    g = reference / reference.norm()
    u = direction / direction.norm() if direction.norm() > 0 else -g
    alignment = u @ g
    if alignment >= minimum_alignment:
        return u
    transverse = u - alignment * g
    if transverse.norm() <= 16 * torch.finfo(u.dtype).eps:
        basis = torch.zeros_like(g)
        basis[torch.argmin(g.abs())] = 1
        transverse = basis - basis @ g * g
    v = transverse / transverse.norm()
    return minimum_alignment * g + math.sqrt(max(0.0, 1 - minimum_alignment ** 2)) * v

def direction_attack(benign, reference, tau=0.5, degree=4, seed=11, round_index=51, steps=50, restarts=5, scale=1.0):
    if not 0 < tau <= 1 or degree < 1 or steps < 0 or (restarts < 1) or (scale < 1):
        raise ValueError('invalid direction attack parameters')
    if benign.ndim != 2 or len(benign) == 0 or (not torch.isfinite(benign).all()):
        raise ValueError('finite nonempty benign update matrix required')
    mean = benign.mean(0)
    if mean.norm() <= 0:
        raise ValueError('normalized benign mean is undefined for a zero mean')
    target = mean / mean.norm()
    h = 2 * tau ** (1 / degree) - 1
    gen = torch.Generator(device=reference.device).manual_seed(100000 * seed + round_index)
    best, best_value = (None, -float('inf'))
    for _ in range(restarts):
        u = project_trust_cap(torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=gen), reference, h)
        for step in range(steps + 1):
            value = float((u - target).square().sum())
            if value > best_value:
                best, best_value = (u.clone(), value)
            if step < steps:
                u = project_trust_cap(u + 0.05 * 2 * (u - target), reference, h)
    return (scale * best, best_value)
