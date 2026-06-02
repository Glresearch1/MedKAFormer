# Contributing

Thanks for your interest in improving MedKAFormer.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Pull Requests

Before opening a pull request:

- Keep changes focused and avoid committing datasets, checkpoints, or logs.
- Run a syntax check with `python3 -m py_compile train_kvasir.py medkanformer.py kvasir/kvasir_dataset.py kvasir/kvasir_utils.py`.
- Document new training options or model variants in `README.md`.

## Reporting Issues

Please include:

- Operating system, Python version, PyTorch version, and GPU/CUDA details.
- The command you ran.
- The full error message or relevant log excerpt.
