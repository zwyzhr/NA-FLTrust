import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, csv, json, time, hashlib, platform, datetime, math, copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import datasets, transforms, models
from nafltrust.core import Config, aggregate, normalize_rows
from nafltrust.attacks import prepare, joint_attack, direction_attack
from nafltrust.crypto import CKKS
from nafltrust.baselines import classifier_defense
from nafltrust.calibrated_objective import CalibratedObjective
HERE = Path(__file__).resolve().parent

def requires_root(method, attack='none', control='none'):
    return method in ('na', 'na_fltrust', 'fltrust', 'polynomial_safeguard', 'polynomial_clip') or attack in ('direction', 'joint', 'root_only') or control == 'root_only'

def model_for(dataset, kind):
    channels = 1 if dataset in ('mnist', 'fashionmnist') else 3
    side = 64 if dataset == 'tiny-imagenet' else 28 if channels == 1 else 32
    classes = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'tiny-imagenet': 200}[dataset]
    if kind == 'linear':
        return nn.Sequential(nn.Flatten(), nn.Linear(channels * side * side, classes))
    if dataset in ('cifar100', 'tiny-imagenet'):
        net = (models.resnet18 if dataset == 'cifar100' else models.resnet34)(num_classes=classes)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        return net
    first = 32 if channels == 1 else 64
    layers = [nn.Conv2d(channels, first, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(first, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
    width = 64 * (side // 4) ** 2
    if channels == 3:
        layers += [nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        width = 128 * 4 * 4
    hidden = 128 if channels == 1 else 512
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, classes))

def load_data(a):
    tf = transforms.ToTensor()
    if a.dataset == 'tiny-imagenet':
        train = datasets.ImageFolder(str(a.data / 'tiny-imagenet-200/train'), tf)
        test = datasets.ImageFolder(str(a.data / 'tiny-imagenet-200/val_by_class'), tf)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError('Tiny ImageNet class mapping mismatch')
    else:
        cls = {'mnist': datasets.MNIST, 'fashionmnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[a.dataset]
        train = cls(str(a.data), train=True, download=a.download, transform=tf)
        test = cls(str(a.data), train=False, download=a.download, transform=tf)
    return (train, test, np.asarray(train.targets), np.asarray(test.targets))

def stratified(labels, count, rng):
    classes = np.unique(labels)
    if count < len(classes):
        raise ValueError('root must contain at least one example per class')
    order = rng.permutation(classes)
    counts = {c: count // len(classes) + int(i < count % len(classes)) for i, c in enumerate(order)}
    return np.concatenate([rng.choice(np.where(labels == c)[0], counts[c], replace=False) for c in classes])

def partitions(y, reserve, n, alpha, rng):
    mask = np.ones(len(y), bool)
    mask[reserve] = False
    for attempt in range(100):
        parts = [[] for _ in range(n)]
        for c in np.unique(y):
            ids = np.flatnonzero((y == c) & mask)
            rng.shuffle(ids)
            counts = rng.multinomial(len(ids), rng.dirichlet(np.full(n, alpha)))
            for i, v in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
                parts[i].extend(v.tolist())
        if all(parts):
            return ([np.asarray(v, dtype=np.int64) for v in parts], attempt)
    raise ValueError('unable to obtain nonempty Dirichlet partitions in 100 attempts')

def nested_order(labels, seed):
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    queues = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in classes]
    return np.asarray([q[j] for j in range(max(map(len, queues))) for q in queues if j < len(q)], dtype=np.int64)

def make_splits(y, ty, root_size, num_clients, alpha, seed):
    if not 1 <= root_size <= 1000:
        raise ValueError('root_size must lie in [1,1000]')
    rng = np.random.default_rng(seed)
    base = stratified(ty, 500, rng)
    reserve = stratified(y, 500, rng)
    parts, retries = partitions(y, reserve, num_clients, alpha, rng)
    test = np.setdiff1d(np.arange(len(ty)), base)
    pool = np.concatenate((base, len(ty) + reserve))
    order = np.concatenate((nested_order(ty[base], seed + 8101), 500 + nested_order(y[reserve], seed + 8102)))
    root = pool if root_size == 1000 else base if root_size == 500 else pool[order[:root_size]]
    return (root, base, reserve, parts, test, retries)

def arguments():
    p = argparse.ArgumentParser(description="NA-FLTrust federated training")
    checks = p.add_argument_group("Functional checks (explicit reduced configurations)")
    p.add_argument('--preset', choices=['original', 'paper'], default='paper')
    p.add_argument('--client_workers', type=int, default=None, help='concurrent independent client trainers; paper defaults to selected_clients')
    p.add_argument('--resume_checkpoint', type=Path)
    p.add_argument('--control', choices=['none', 'frozen', 'root_only'], default='none')
    p.add_argument('--checkpoint_round', type=int, default=50)
    p.add_argument('--stop_after_round', type=int, default=0)
    p.add_argument('--audit_workers', type=int, default=2, help='maximum concurrent independent committee audits')
    checks.add_argument('--unit_degree_diagnostic', action='store_true', help='explicit plaintext unit-input k=1 diagnostic, not the encrypted protocol')
    p.add_argument('--release_budget', type=int, default=2 ** 40)
    p.add_argument('--dataset', choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'tiny-imagenet'], default='mnist')
    p.add_argument('--krum_policy', choices=['strict', 'paper_stress'], default=None)
    p.add_argument('--fusion', choices=['na', 'na_fltrust', 'fltrust', 'fedavg', 'average', 'median', 'krum', 'trimmed_mean', 'dual_defense', 'cos_defense', 'clipping_median', 'fedavg_weighted', 'polynomial_safeguard', 'polynomial_clip'], default='na')
    p.add_argument('--backend', choices=['plaintext', 'ckks-smoke', 'ckks-paper', 'ckks-paper-wide'], default=None, help='default: ckks-paper for NA-FLTrust, plaintext for other methods')
    p.add_argument('--num_clients', type=int, default=100)
    p.add_argument('--selected_clients', type=int, default=10)
    p.add_argument('--training_round', type=int, default=None)
    p.add_argument('--local_epochs', type=int, default=None)
    p.add_argument('--optimizer', choices=['sgd'], default='sgd')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--attacker_strategy', choices=['none', 'model_poisoning_ipm', 'scale', 'alie', 'label_flip', 'joint', 'root_only', 'direction'], default='model_poisoning_ipm')
    p.add_argument('--attack_start_round', type=int, default=None, help='first attacked round, 1-based')
    p.add_argument('--attacker_ratio', type=float, default=0.3)
    p.add_argument('--epsilon', type=float, default=0.01)
    p.add_argument('--attack_scale', type=float, default=10.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--degree', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--eta', type=float, default=1.0)
    p.add_argument('--root_size', type=int, default=500)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--device', default='cuda')
    checks.add_argument('--smoke_eval_records', type=int, default=0, help='explicit pipeline-test subset of the fixed test pool; zero evaluates all')
    p.add_argument('--model', choices=['paper', 'linear'], default='paper')
    checks.add_argument('--smoke_batches', type=int, default=0)
    p.add_argument('--joint_steps', type=int, default=200)
    p.add_argument('--joint_restarts', type=int, default=10)
    p.add_argument('--direction_reference', choices=['proxy', 'partial', 'full'], default='proxy')
    p.add_argument('--direction_tau', type=float, default=0.5)
    p.add_argument('--direction_scale', type=float, default=1.0)
    p.add_argument('--direction_steps', type=int, default=50)
    p.add_argument('--direction_restarts', type=int, default=5)
    p.add_argument('--data', type=Path, default=HERE / 'data')
    p.add_argument('--download', action='store_true')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--holders', type=int, default=None)
    p.add_argument('--threshold', type=int, default=None)
    a = p.parse_args()
    a.fusion = 'na' if a.fusion == 'na_fltrust' else 'fedavg' if a.fusion == 'average' else a.fusion
    if a.backend is None:
        a.backend = 'ckks-paper' if a.fusion == 'na' else 'plaintext'
    paper = a.preset == 'paper'
    if a.client_workers is None:
        a.client_workers = a.selected_clients if paper else 1
    if min(a.client_workers, a.audit_workers) < 1:
        p.error('worker counts must be positive')
    defaults = dict(training_round=200 if paper else 100, local_epochs=5 if paper else 3, attack_start_round=51 if paper else 50, learning_rate=(0.1 if a.dataset not in ('mnist', 'fashionmnist') else 0.01) if paper else 0.01, holders=20 if paper else 3, threshold=14 if paper else 2)
    for key, value in defaults.items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    if a.krum_policy is None:
        a.krum_policy = 'paper_stress' if paper else 'strict'
    if a.attacker_strategy == 'root_only' and a.fusion != 'na':
        p.error('root-only attack construction targets NA-FLTrust')
    if a.control != 'none' and a.resume_checkpoint is None:
        p.error('controls require a paired checkpoint')
    if a.checkpoint_round < 1 or a.stop_after_round < 0 or a.release_budget < 1:
        p.error('invalid checkpoint or budget')
    if not 0 <= a.attacker_ratio < 1 or not 1 <= a.selected_clients <= a.num_clients or a.alpha <= 0:
        p.error('invalid participation/attack parameters')
    if not 0 <= a.smoke_eval_records <= 9500 or (a.smoke_eval_records and (not a.smoke_batches)):
        p.error('smoke_eval_records requires smoke_batches and must be in [0,9500]')
    if min(a.training_round, a.local_epochs, a.batch_size, a.root_size) <= 0 or a.smoke_batches < 0:
        p.error('training counts must be positive')
    if a.backend.startswith('ckks-') and (a.degree < 2 or a.degree % 2 or a.degree > (16 if a.backend == 'ckks-paper-wide' else 4)):
        p.error('degree is outside the selected CKKS profile')
    if a.unit_degree_diagnostic and (a.backend != 'plaintext' or a.attacker_strategy in ('joint', 'root_only', 'scale') or a.direction_scale > 1):
        p.error('unit diagnostic requires plaintext unit inputs')
    if a.backend.startswith('ckks-') and a.fusion != 'na':
        p.error('CKKS backend implements NA-FLTrust only')
    if a.backend == 'ckks-paper-wide' and (a.attacker_strategy in ('joint', 'root_only') or (a.attacker_strategy == 'scale' and a.attack_scale > 1) or (a.attacker_strategy == 'direction' and a.direction_scale > 1)):
        p.error('paper-wide is a unit-input degree profile; amplified attacks require ckks-paper with k<=4')
    if a.attacker_strategy == 'joint' and a.fusion not in ('na', 'fltrust', 'polynomial_safeguard', 'polynomial_clip'):
        p.error('joint/root-only controls require trust aggregation')
    if a.degree != 4 and a.attacker_strategy == 'root_only':
        p.error('root-only construction uses k=4')
    if not 0 < a.direction_tau <= 1 or a.direction_scale < 1 or a.direction_steps < 0 or (a.direction_restarts < 1):
        p.error('invalid direction attack configuration')
    return a

def main():
    a = arguments()
    a.out = a.out or HERE / 'runs' / f'{a.dataset}_{a.fusion}_{a.backend}_{a.attacker_strategy}_{a.seed}_{time.time_ns()}'
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(a.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    cfg = Config(a.degree, a.beta, a.kappa, a.eta, unit_diagnostic=a.unit_degree_diagnostic)
    train, test, y, ty = load_data(a)
    rng = np.random.default_rng(a.seed)
    classes = len(np.unique(y))
    root_ids, base_root, reserve, parts, test_ids, retries = make_splits(y, ty, a.root_size, a.num_clients, a.alpha, a.seed)
    root_data = torch.utils.data.ConcatDataset([test, train])
    root_labels = np.concatenate((ty, y))
    if a.smoke_eval_records:
        test_ids = test_ids[nested_order(ty[test_ids], a.seed + 8801)[:a.smoke_eval_records]]
    assert not set(base_root) & set(test_ids)
    joined = np.concatenate(parts)
    assert len(joined) == len(np.unique(joined)) == len(y) - len(reserve)
    split_values = dict(root=root_ids, base_root=base_root, test=test_ids, reserve=reserve, **{f'client_{i}': v for i, v in enumerate(parts)})
    np.savez_compressed(a.out / 'splits.npz', **split_values)
    model = model_for(a.dataset, a.model).to(device)
    worker = model_for(a.dataset, a.model).to(device)
    manifest = {**vars(a), 'data': str(a.data), 'out': str(a.out), 'resume_checkpoint': str(a.resume_checkpoint) if a.resume_checkpoint else None, 'torch': torch.__version__, 'platform': platform.platform(), 'parameters': sum((p.numel() for p in model.parameters())), 'partition_rejections': retries, 'test_records': len(test_ids), 'root_source': 'fixed 500 evaluation root plus disjoint training reserve for roots above 500', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'running', 'kind': 'pipeline_smoke' if a.smoke_batches else 'executed_training', 'code_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE / 'main.py', *sorted((HERE / 'nafltrust').glob('*.py')), *sorted((HERE / 'crypto').glob('*.go'))]}, 'batchnorm_policy': 'persistent client-local buffers; disposable inference copy reset and calibrated on ordered training reserve with cumulative averaging'}

    def save_manifest():
        (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    manifest['root_required_by_aggregation'] = requires_root(a.fusion, control=a.control)
    manifest['root_reference_computed'] = requires_root(a.fusion, a.attacker_strategy, a.control)
    save_manifest()
    crypto = None

    def batches(ds, ids, gen=None, shuffle=False, batch_size=None):
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ids.tolist()), batch_size=batch_size or a.batch_size, shuffle=shuffle, generator=gen, num_workers=0)
        yield from loader
    local_buffers = {}
    first_round = 1
    checkpoint_keys = ('dataset', 'model', 'seed', 'num_clients', 'selected_clients', 'alpha', 'root_size', 'learning_rate', 'training_round', 'local_epochs', 'batch_size', 'smoke_batches', 'smoke_eval_records')
    if a.resume_checkpoint:
        state = torch.load(a.resume_checkpoint, map_location=device, weights_only=False)
        for key in checkpoint_keys:
            if state['config'].get(key, 0 if key == 'smoke_eval_records' else None) != getattr(a, key):
                raise ValueError(f'checkpoint configuration mismatch: {key}')
        for key, value in split_values.items():
            if not np.array_equal(state['splits'][key], value):
                raise ValueError(f'checkpoint split mismatch: {key}')
        model.load_state_dict(state['model'])
        local_buffers = state['local_buffers']
        first_round = state['round'] + 1
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        if first_round > min(a.training_round, a.stop_after_round or a.training_round):
            raise ValueError('checkpoint has no remaining requested rounds')

    def checkpoint(rr):
        torch.save(dict(round=rr, config={key: getattr(a, key) for key in checkpoint_keys}, model=model.state_dict(), local_buffers=local_buffers, splits=split_values, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), a.out / f'checkpoint_{rr}.pt')

    def train_delta(start, ids, ds, rr, client, flip=False):
        worker = copy.deepcopy(model)
        worker.load_state_dict(model.state_dict())
        vector_to_parameters(start.clone(), worker.parameters())
        if client in local_buffers:
            with torch.no_grad():
                for name, buffer in worker.named_buffers():
                    buffer.copy_(local_buffers[client][name])
        worker.train()
        lr = a.learning_rate * (1 + math.cos(math.pi * (rr - 1) / a.training_round)) / 2 if a.dataset == 'tiny-imagenet' else a.learning_rate
        opt = torch.optim.SGD(worker.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(a.seed * 10000000 + rr * 10000 + int(client) + 1)
        for ep in range(a.local_epochs):
            for j, (x, labels) in enumerate(batches(ds, ids, gen, True)):
                if a.smoke_batches and j >= a.smoke_batches:
                    break
                x, labels = (x.to(device), labels.to(device))
                if flip:
                    labels = (labels + 1) % classes
                opt.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(worker(x), labels).backward()
                opt.step()
        local_buffers[client] = {name: buffer.detach().clone() for name, buffer in worker.named_buffers()}
        return parameters_to_vector(worker.parameters()).detach() - start

    def evaluate():
        evaluation = model
        if any((isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())):
            evaluation = copy.deepcopy(model)
            for layer in evaluation.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.reset_running_stats()
                    layer.momentum = None
            evaluation.train()
            with torch.no_grad():
                for x, _ in batches(train, np.sort(reserve), batch_size=64):
                    evaluation(x.to(device))
        evaluation.eval()
        pred = []
        with torch.no_grad():
            for x, _ in batches(test, test_ids, batch_size=64):
                pred.append(evaluation(x.to(device)).argmax(1).cpu())
        return torch.cat(pred).numpy()
    fields = ['round', 'accuracy', 'macro_f1', 'correct', 'n_test', 'accepted', 'seconds', 'selected', 'malicious_count', 'attack_loss']
    try:
        crypto = CKKS(a.backend[5:], a.holders, a.threshold, a.out / 'crypto.log', release_budget=a.release_budget, audit_workers=a.audit_workers) if a.backend.startswith('ckks-') else None
        with (a.out / 'rounds.csv').open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rr in range(first_round, min(a.training_round, a.stop_after_round or a.training_round) + 1):
                start_time = time.perf_counter()
                start = parameters_to_vector(model.parameters()).detach().clone()
                selected = np.random.default_rng(a.seed * 100000 + rr).choice(a.num_clients, a.selected_clients, False)
                expected = a.attacker_ratio * a.selected_clients
                b = math.floor((rr - a.attack_start_round + 1) * expected + 1e-09) - math.floor((rr - a.attack_start_round) * expected + 1e-09) if rr >= a.attack_start_round and a.attacker_strategy != 'none' and (a.control == 'none') else 0
                if b >= a.selected_clients:
                    raise ValueError('selected round has no benign clients')
                root_required = requires_root(a.fusion, control=a.control)
                reference_required = requires_root(a.fusion, a.attacker_strategy, a.control)
                gs = train_delta(start, root_ids, root_data, rr, a.num_clients + 1).double() if reference_required else torch.zeros_like(start, dtype=torch.float64)
                valid_root = bool(torch.isfinite(gs).all() and torch.isfinite(gs.norm()) and (gs.norm() > 0))

                def train_selected(item):
                    j, c = item
                    return train_delta(start, parts[c], train, rr, c, flip=a.attacker_strategy == 'label_flip' and j >= len(selected) - b)
                if root_required and (not valid_root):
                    raw = torch.zeros((len(selected), len(gs)), dtype=gs.dtype, device=gs.device)
                else:
                    with ThreadPoolExecutor(max_workers=min(a.client_workers, len(selected))) as pool:
                        raw = torch.stack(list(pool.map(train_selected, enumerate(selected)))).double()
                attack = {'model_poisoning_ipm': 'ipm'}.get(a.attacker_strategy, a.attacker_strategy)
                received = prepare(raw, b, a.fusion, attack, a.epsilon, a.attack_scale)
                loss = None
                if b and attack == 'direction' and valid_root:
                    if a.direction_reference == 'full':
                        reference = gs
                        reference_ids = root_ids
                        reference_source = 'root'
                    elif a.direction_reference == 'partial':
                        subset = nested_order(root_labels[root_ids], a.seed + 90210)[:min(250, len(root_ids))]
                        reference_ids = root_ids[subset]
                        reference_source = 'root'
                        reference = train_delta(start, reference_ids, root_data, rr, a.num_clients + 2).double()
                    else:
                        available = np.concatenate([parts[c] for c in selected[-b:]])
                        if len(available) < 500:
                            raise ValueError('direction proxy requires 500 distinct attacker-training records')
                        reference_ids = np.random.default_rng(a.seed * 100000 + rr + 90210).choice(available, 500, replace=False)
                        reference_source = 'attacker_training'
                        reference = train_delta(start, reference_ids, train, rr, a.num_clients + 3).double()
                    with (a.out / 'direction_reference.jsonl').open('a', encoding='utf8') as reference_log:
                        reference_log.write(json.dumps(dict(round=rr, source=reference_source, indices=reference_ids.tolist())) + '\n')
                    x, loss = direction_attack(received[:-b], reference, a.direction_tau, a.degree, a.seed, rr, a.direction_steps, a.direction_restarts, a.direction_scale)
                    received[-b:] = x
                if b and attack == 'joint' and valid_root:
                    idx = np.concatenate([parts[c] for c in selected[-b:]])[:500]
                    xs, ys = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(train, idx.tolist()), batch_size=len(idx))))
                    if len(idx) != 500:
                        raise ValueError('joint attack requires 500 distinct attacker-training records')
                    xs, ys = (xs.to(device), ys.to(device))
                    calibration = [x.to(device) for x, _ in batches(train, np.sort(reserve), batch_size=64)] if any((isinstance(m, nn.BatchNorm2d) for m in model.modules())) else []
                    objective = CalibratedObjective(model, start, calibration, xs, ys)
                    manifest['joint_objective'] = 'candidate-specific cumulative BN calibration on ordered training reserve, differentiated through calibration; exact rounded aggregation forward'
                    x, loss = joint_attack(received[:-b], gs, b, objective, a.fusion, cfg, a.seed, rr, a.joint_steps, a.joint_restarts)
                    received[-b:] = x
                if b and attack == 'root_only' and valid_root:
                    g = normalize_rows(gs)
                    ben = received[:-b]
                    H = ((ben @ g + 1) / 2) ** 4 @ ben
                    received[-b:] = -15 * g - (H - H @ g * g) / (2401 * b)
                if root_required and (not valid_root):
                    delta = torch.zeros_like(gs)
                    ok = False
                elif a.control == 'frozen':
                    delta = torch.zeros_like(gs)
                    ok = True
                elif a.control == 'root_only':
                    delta = cfg.eta * cfg.beta * gs
                    ok = True
                else:
                    if a.fusion in ('dual_defense', 'cos_defense'):
                        delta, ok = classifier_defense(start, received, model, [len(parts[c]) for c in selected], a.fusion, a.seed, rr, a.epsilon)
                    elif a.fusion == 'clipping_median':
                        delta = (start.double() + received.quantile(0.5, dim=0)).clamp(-0.1, 0.1) - start.double()
                        ok = True
                    elif a.fusion == 'fedavg_weighted':
                        weights = torch.tensor([len(parts[c]) for c in selected], dtype=received.dtype, device=device)
                        weights /= weights.sum()
                        delta = weights @ received
                        ok = True
                    else:
                        delta, ok = crypto.aggregate(received, gs, cfg, client_ids=selected.tolist()) if crypto else aggregate(received, gs, a.fusion, cfg, min(b, (len(received) - 3) // 2) if a.fusion == 'krum' and a.krum_policy == 'paper_stress' else b)
                vector_to_parameters(start + delta.to(start.dtype), model.parameters())
                pred = evaluate()
                truth = ty[test_ids]
                confusion = np.bincount(classes * truth + pred, minlength=classes ** 2).reshape(classes, classes)
                tp = np.diag(confusion)
                precision = np.divide(tp, confusion.sum(0), out=np.zeros(classes), where=confusion.sum(0) > 0)
                recall = np.divide(tp, confusion.sum(1), out=np.zeros(classes), where=confusion.sum(1) > 0)
                f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=precision + recall > 0)
                row = dict(round=rr, accuracy=100 * float((pred == truth).mean()), macro_f1=float(f1.mean()), correct=int((pred == truth).sum()), n_test=len(truth), accepted=int(ok), seconds=time.perf_counter() - start_time, selected=json.dumps(selected.tolist()), malicious_count=b, attack_loss=loss)
                writer.writerow(row)
                f.flush()
                print(json.dumps(row), flush=True)
                if rr == a.checkpoint_round:
                    checkpoint(rr)
            np.savez_compressed(a.out / 'predictions.npz', ids=test_ids, labels=truth, predictions=pred, confusion=confusion, precision=precision, recall=recall, f1=f1)
            torch.save(model.state_dict(), a.out / 'model.pt')
            checkpoint(rr)
            manifest.update(status='completed' if rr == a.training_round else 'checkpointed', completed_rounds=rr)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        if crypto:
            crypto.close()
            (a.out / 'crypto_metrics.json').write_text(json.dumps(crypto.metrics, indent=2), encoding='utf8')
        manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_manifest()
if __name__ == '__main__':
    main()
