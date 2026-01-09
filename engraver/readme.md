## Create venv

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## PIP

### Remove pip packages

```bash
pip freeze | % { pip uninstall -y ($_ -split '==')[0] }
```

### List pip packages
```bash
pip list
```

### Install pip packages
```bash
pip install -r requirements.txt
```

### Install tkinter
```bash
sudo apt install python3-tk
```


## Pytest

### How to use
1. Put your implementation in a module importable as `shapefit`
   (or edit `tests/conftest.py` to point at your module name).
   
2. Run tests:
   ```bash
   pip install -e .
   pip install pytest
   pytest


## ruff

```bash
ruff check .
```