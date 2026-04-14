import json
from typing import List

from config import SPLIT_STR, SYSTEM_PROMPT
from coze_service import CozeService
from models.schema import ClinicalReport
from utils.file_reader import read_all_txt_files_in_folder
from utils.report_parser import parse_clinical_reports


def build_dataset(reports: List[ClinicalReport], coze_service: CozeService):
    dataset = []
    for i, report in enumerate(reports, start=1):
        print(
            f'\nProcessing report {i} (source: {report.source_file}): '
            f'{report.report_content[:50]}...'
        )
        answer = coze_service.get_answer(report.report_content)
        if '文中未提及' in answer:
            print(f'Report {i} skipped due to unsupported answer')
            continue

        dataset.append(
            {
                'instruction': SYSTEM_PROMPT,
                'input': report.report_content,
                'output': answer,
                'source_file': report.source_file,
            }
        )
        print(f'Report {i} processed')
    return dataset


def run_pipeline(question_folder_path: str, output_json_path: str, coze_service: CozeService):
    all_file_contents, all_file_names = read_all_txt_files_in_folder(question_folder_path)
    reports = parse_clinical_reports(all_file_contents, all_file_names, SPLIT_STR)
    dataset = build_dataset(reports, coze_service)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print('\n===== Final Summary =====')
    print(f'Total files: {len(all_file_names)}')
    print(f'Total reports extracted: {len(reports)}')
    print(f'Final dataset size: {len(dataset)}')
    print(f'Saved to: {output_json_path}')
