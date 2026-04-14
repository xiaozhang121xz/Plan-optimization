from cozepy import COZE_CN_BASE_URL
from cozepy import ChatEventType, Coze, Message, TokenAuth


class CozeService:
    def __init__(self, api_token: str, bot_id: str, user_id: str):
        self.bot_id = bot_id
        self.user_id = user_id
        self.client = Coze(
            auth=TokenAuth(token=api_token),
            base_url=COZE_CN_BASE_URL,
        )

    def get_answer(self, question: str) -> str:
        full_answer = ''
        try:
            for event in self.client.chat.stream(
                bot_id=self.bot_id,
                user_id=self.user_id,
                additional_messages=[Message.build_user_question_text(question)],
            ):
                if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                    full_answer += event.message.content
        except Exception as e:
            print(f'??????????{question[:20]}... ?????{e}')
            return ''
        return full_answer
