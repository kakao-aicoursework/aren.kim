import pynecone as pc
import os
from datetime import datetime
from pynecone.base import Base
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader

f = open("./apiKey.txt", 'r')
apiKey = f.readline()
f.close()
os.environ["OPENAI_API_KEY"] = apiKey

loader = TextLoader("./project_data_kakaosync.txt")
loader.load()

llm = ChatOpenAI(temperature=0.8, model_name='gpt-3.5-turbo-16k')


def makePromptTemplate() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["text"],
        template="당신은 최고의 상담봇 입니다.\n 사용자의 질문에 대한 적절한 답변을 줍니다.\n <질문>: {text}"
    )


def askKakaoSync(text) -> str:
    prompt_template = makePromptTemplate()
    chain = LLMChain(llm=llm, verbose=True, prompt=prompt_template)
    return chain.run(text)


class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    answer = "Answer will appear here."

    def output(self):
        if not self.text.strip():
            return "Answer will appear here."
        self.answer = askKakaoSync(self.text)

    def post(self):
        self.output()
        self.messages = [
                            Message(
                                original_text=self.text,
                                text=self.answer,
                                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                            )
                        ] + self.messages

def header():
    return pc.box(
        pc.text("SyncBotHelper 📻", font_size="2rem"),
    )

def message(message):
    return pc.box(
        pc.vstack(
            pc.text("<질문>: "+ message.original_text, font_weight="bold"),
            pc.text((message.text)),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )

def index():
    return pc.container(
        header(),
        pc.input(
            placeholder="SyncHelper에게 물어보세요!",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="ChatBot")
app.compile()