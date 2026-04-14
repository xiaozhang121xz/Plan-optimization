# Dateset Project

This folder is the Python-module version of `shujuji.ipynb`.

## Structure

```text
dateset/
??? src/
?   ??? main.py
?   ??? config.py
?   ??? coze_service.py
?   ??? pipeline.py
?   ??? models/
?   ?   ??? schema.py
?   ??? utils/
?       ??? file_reader.py
?       ??? report_parser.py
??? tests/
??? requirements.txt
??? setup.py
??? README.md
```

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

## Environment variables

- `COZE_API_TOKEN` (required)
- `COZE_BOT_ID` (optional, default from notebook)
- `COZE_USER_ID` (optional)
- `QUESTION_FOLDER_PATH` (optional)
- `OUTPUT_JSON_PATH` (optional)
