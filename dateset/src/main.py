import os

from config import (
    DEFAULT_BOT_ID,
    DEFAULT_OUTPUT_JSON_PATH,
    DEFAULT_QUESTION_FOLDER_PATH,
    DEFAULT_USER_ID,
)
from coze_service import CozeService
from pipeline import run_pipeline


def main() -> None:
    api_token = os.getenv('COZE_API_TOKEN')
    if not api_token:
        raise ValueError(' COZE_API_TOKEN')

    coze_service = CozeService(
        api_token=api_token,
        bot_id=DEFAULT_BOT_ID,
        user_id=DEFAULT_USER_ID,
    )

    run_pipeline(
        question_folder_path=DEFAULT_QUESTION_FOLDER_PATH,
        output_json_path=DEFAULT_OUTPUT_JSON_PATH,
        coze_service=coze_service,
    )


if __name__ == '__main__':
    main()
