from pathlib import Path
import json, subprocess, torch, uuid, hashlib, os, tempfile
from concurrent.futures import ThreadPoolExecutor
from .core import Config, release

def send_json(stream, value):
    payload = json.dumps(value, allow_nan=False) + '\n'
    for offset in range(0, len(payload), 65536):
        stream.write(payload[offset:offset + 65536])
    stream.flush()

class RoundRejected(RuntimeError):

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

class CKKS:

    def __init__(self, profile='paper', holders=20, threshold=14, log=None, slots=0, release_budget=2 ** 40, audit_workers=2):
        name = 'nafltrust-ckks.exe' if os.name == 'nt' else 'nafltrust-ckks'
        exe = Path(os.environ['NAFLTRUST_BACKEND_EXE']).resolve() if os.environ.get('NAFLTRUST_BACKEND_EXE') else Path(__file__).resolve().parents[1] / 'bin' / name
        if not exe.exists():
            raise FileNotFoundError('Build the Go backend first: see README.md')
        self.log = open(log, 'w', encoding='utf8') if log else None
        self.command = [str(exe), '--profile', profile, '--holders', str(holders), '--threshold', str(threshold), '--slots', str(slots), '--release-budget', str(release_budget), '--audit-workers', str(audit_workers)]
        self._start()
        self.metrics = []
        cache_root = Path(__file__).resolve().parents[1] / '.runtime_cache'
        cache_root.mkdir(exist_ok=True)
        self.archive = tempfile.TemporaryDirectory(prefix='nafltrust-results-', dir=cache_root)
        self.completed = {}
        self.aborted = set()
        self.epoch_rotations = 0
        self.needs_refresh = False

    def _start(self):
        self.clients = []
        self.client_ids = []
        self.client_keys = []
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf8', bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            self._stop_process()
            raise RuntimeError('backend setup failed; inspect crypto log')
        self.public_setup = json.loads(line)
        if not self.public_setup.get('ready'):
            raise RuntimeError('invalid public setup')
        self.clients = []
        self.client_ids = []
        self.client_keys = []

    def _encrypt(self, values, session, client_ids=None):
        ids = list(range(len(values))) if client_ids is None else list(map(int, client_ids))
        if len(ids) != len(values) or len(set(ids)) != len(ids) or any((i < 0 for i in ids)):
            raise ValueError('invalid client identities')
        changed = False
        for identity in ids:
            if identity in self.client_ids:
                continue
            child = subprocess.Popen([self.command[0], '--client-worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf8', bufsize=1)
            self.clients.append(child)
            self.client_ids.append(identity)
            send_json(child.stdin, dict(setup=self.public_setup, id=identity))
            key = json.loads(child.stdout.readline())['identity']
            while len(self.client_keys) <= identity:
                self.client_keys.append(None)
            self.client_keys[identity] = key
            changed = True
        if changed:
            send_json(self.proc.stdin, dict(client_keys=self.client_keys))
            response = json.loads(self.proc.stdout.readline())
            if not response.get('registered'):
                raise RuntimeError(response.get('error', 'identity registration failed'))

        def encrypt(item):
            identity, vector = item
            child = self.clients[self.client_ids.index(identity)]
            send_json(child.stdin, dict(round=session, values=vector))
            line = child.stdout.readline()
            if not line:
                raise RuntimeError('client encryption failed')
            submission = json.loads(line)
            if submission.get('error'):
                raise ValueError(submission['error'])
            return submission
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            return list(pool.map(encrypt, zip(ids, values)))

    def _stop_process(self):
        for child in self.clients:
            if child.poll() is None:
                child.stdin.close()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)

    def request(self, updates, root, session=None, degree=4, active=None, tamper=False, drop_after=0, client_ids=None):
        req = dict(session=session or uuid.uuid4().hex, updates=updates.detach().cpu().tolist(), root=root.detach().cpu().tolist(), degree=degree, tamper=tamper)
        if client_ids is not None:
            req['client_ids'] = list(map(int, client_ids))
        if drop_after:
            req['drop_after'] = drop_after
        if active is not None:
            req['active'] = active
        payload = json.dumps(req, allow_nan=False)
        binding = hashlib.sha256(payload.encode()).hexdigest()
        sid = req['session']
        if sid in self.aborted:
            raise RoundRejected('aborted_session', 'session previously aborted; use fresh round and ciphertexts')
        if sid in self.completed:
            previous, cache_path = self.completed[sid]
            cached = json.loads(Path(cache_path).read_text())
            if previous != binding:
                raise RuntimeError('session already bound to different input or active set')
            result = {**cached, 'cached': True}
            self.metrics.append({k: v for k, v in result.items() if k not in ['numerator', 'denominator']})
            return result
        if self.needs_refresh:
            self._stop_process()
            self._start()
            self.epoch_rotations += 1
            self.needs_refresh = False
        for attempt in range(2):
            wire = {k: v for k, v in req.items() if k not in ('updates', 'client_ids')}
            wire['encrypted'] = self._encrypt(req['updates'], sid, req.get('client_ids'))
            send_json(self.proc.stdin, wire)
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f'CKKS process exited ({self.proc.poll()}); inspect crypto log')
            result = json.loads(line)
            if result.get('error_code') == 'budget_exhausted' and attempt == 0:
                self.metrics.append(result)
                self._stop_process()
                self._start()
                self.epoch_rotations += 1
                continue
            break
        if result.get('error'):
            self.metrics.append(result)
            if result.get('error_code') in ('audit_rejected', 'holder_dropout', 'insufficient_holders'):
                self.aborted.add(sid)
                if result['error_code'] == 'holder_dropout':
                    self.needs_refresh = True
                raise RoundRejected(result['error_code'], result['error'])
            raise RuntimeError(result['error'])
        cache_path = Path(self.archive.name) / (hashlib.sha256(sid.encode()).hexdigest() + '.json')
        cache_path.write_text(json.dumps(result))
        self.completed[sid] = (binding, str(cache_path))
        self.metrics.append({k: v for k, v in result.items() if k not in ['numerator', 'denominator']})
        return result

    def aggregate(self, updates, root, cfg=Config(), **faults):
        if not torch.isfinite(root).all() or not torch.isfinite(root.norm()) or root.norm() == 0:
            return (torch.zeros_like(root), False)
        try:
            r = self.request(updates, root, degree=cfg.degree, **faults)
        except RoundRejected:
            return (torch.zeros_like(root), False)
        return release(torch.tensor(r['numerator'], dtype=root.dtype, device=root.device), torch.tensor(r['denominator'], dtype=root.dtype, device=root.device), root, cfg)

    def close(self):
        self._stop_process()
        if self.log:
            self.log.close()
        self.archive.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
