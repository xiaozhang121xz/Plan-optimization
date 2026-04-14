from dataclasses import dataclass


@dataclass
class ClinicalReport:
    report_content: str
    source_file: str
