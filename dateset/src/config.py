import os

SPLIT_STR = '???DVH????'

SYSTEM_PROMPT = '''

'''

DEFAULT_BOT_ID = os.getenv('COZE_BOT_ID', ' ')
DEFAULT_USER_ID = os.getenv('COZE_USER_ID', ' ')
DEFAULT_QUESTION_FOLDER_PATH = os.getenv(
    'QUESTION_FOLDER_PATH',
    ' ',
)
DEFAULT_OUTPUT_JSON_PATH = os.getenv(
    'OUTPUT_JSON_PATH',
    ' ',
)
