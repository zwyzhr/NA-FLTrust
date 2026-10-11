# NA-FLTrust: Norm-Aware Trust Aggregation for Byzantine-Robust Federated Learning under Homomorphic Encryption

Norm-aware encrypted aggregation with polynomial trust and a root-direction safeguard.

## Source

- `nafltrust/core.py`: trust weights, norm-aware aggregation, and root-direction safeguard.
- `nafltrust/crypto.py`: client encryption and threshold-release interface.
- `crypto/`: CKKS circuit, committee roles, commitment checks, and release policy.
- `nafltrust/attacks.py`: poisoning and adaptive attacks.
- `nafltrust/baselines.py`: plaintext comparison methods.
- `main.py`: federated training.

## Setup

Install Python dependencies and Go 1.24 or later. Install a CUDA-compatible PyTorch build when using a GPU.

```sh
python -m pip install -r requirements.txt
python build.py
```

## Training

```sh
python main.py --preset paper --backend ckks-paper --holders 20 --threshold 14 --num_clients 100 --dataset mnist --fusion na --training_round 200 --local_epochs 5 --optimizer sgd --batch_size 64 --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 51 --attacker_ratio 0.3 --epsilon 0.01 --device cuda --download
```

## Method correspondence

| Paper component | Implementation |
|---|---|
| Polynomial trust and norm-weighted denominator | `nafltrust/core.py`: `encrypted_inputs_reference` |
| Root-direction safeguard and rounded release | `nafltrust/core.py`: `safeguard`, `release` |
| Packed encrypted trust, norms and weighted sums | `crypto/main.go`: `trustAndNorm`, `terms` |
| Committee setup, threshold decryption and aggregate reconstruction | `crypto/roles.go`, `crypto/main.go` |
| Committed sampled audit | `crypto/auditor.go`, `crypto/challenge.go`, `crypto/merkle.go` |
| Session binding and release budget | `crypto/release_policy.go`, `crypto/roles.go`, `nafltrust/crypto.py` |
| Direction-scale attack | `nafltrust/attacks.py`: `joint_attack` |
| Local training and global update | `main.py` |
