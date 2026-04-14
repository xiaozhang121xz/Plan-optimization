# Project Overview

This repository currently contains three Python workflows:

1. `src/` (main radiation-planning script)
2. `sft/` (SFT training/inference project converted from `FSFT.ipynb`)
3. `dateset/` (dataset generation project converted from `shujuji.ipynb`)

## Directory Structure

```text
.
├── src/                  # Main script (includes MATLAB engine usage)
├── sft/                  # SFT subproject
├── dateset/              # Dataset-building subproject
├── matlab/               # MATLAB helper scripts
├── tests/
├── requirements.txt      # Root shared dependencies
├── setup.py
└── README.md
```

## Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Guide

### 1) Root `src` workflow

```bash
python src/main.py
```

### 2) `sft` workflow

```bash
python sft/src/main.py --mode train
python sft/src/main.py --mode infer
```

### 3) `dateset` workflow

```bash
set COZE_API_TOKEN=your_token
python dateset/src/main.py
```

## Notes

- `dateset/src/main.py` requires environment variable `COZE_API_TOKEN`.
- `src/main.py` imports `matlab.engine`; ensure MATLAB Engine for Python is installed in your environment.
- Original notebooks (`FSFT.ipynb`, `shujuji.ipynb`, etc.) are kept for reference.
