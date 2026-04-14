# SFT Subproject

`FSFT.ipynb` has been split into Python modules under `sft/src`.

## Structure

```text
sft/
??? src/
?   ??? main.py
?   ??? sft_train.py
?   ??? sft_infer.py
?   ??? models/
?   ??? utils/
?   ??? __init__.py
??? tests/
??? notebooks/
?   ??? FSFT.ipynb
??? requirements.txt
??? setup.py
??? .gitignore
??? README.md
```

## Run

```bash
pip install -r requirements.txt
python src/main.py --mode train
python src/main.py --mode infer
```

## Environment Variables

- `SFT_MODEL_NAME`
- `SFT_DATASET_PATH`
- `SFT_OUTPUT_DIR`
- `SFT_SAVE_DIR`
- `BASE_MODEL_PATH`
- `LORA_MODEL_PATH`
