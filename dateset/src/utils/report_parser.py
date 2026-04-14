from typing import List

from models.schema import ClinicalReport


def parse_clinical_reports(
    all_file_contents: List[str],
    all_file_names: List[str],
    split_str: str,
) -> List[ClinicalReport]:
    reports: List[ClinicalReport] = []

    for idx, file_content in enumerate(all_file_contents):
        file_name = all_file_names[idx]
        print(f'\nParsing file: {file_name}')

        split_parts = file_content.split(split_str)
        for part in split_parts:
            if not part.strip():
                continue
            full_report = f'{split_str}{part}'.strip()
            if len(full_report) > len(split_str) + 1:
                reports.append(ClinicalReport(report_content=full_report, source_file=file_name))

    print(f'\nExtracted {len(reports)} valid clinical reports')
    return reports
