import os
from typing import List, Tuple


def read_all_txt_files_in_folder(folder_path: str) -> Tuple[List[str], List[str]]:
    all_file_contents = []
    all_file_names = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if not file_name.endswith('.txt'):
                continue
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    all_file_contents.append(content)
                    all_file_names.append(file_path)
                    print(f'Read file: {file_path}')
                else:
                    print(f'Skip empty file: {file_path}')
            except Exception as e:
                print(f'Failed to read {file_path}: {e}')

    print(f'\nLoaded {len(all_file_contents)} valid txt files')
    return all_file_contents, all_file_names
