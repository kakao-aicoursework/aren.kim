import pynecone as pc
import os
from datetime import datetime
from pynecone.base import Base
import json
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

CURRENT_DIR = os.path.dirname(os.getcwd()+"/kakaoAPIChatBot")
f = open(CURRENT_DIR + "/apiKey.txt", 'r')
apiKey = f.readline()
googleApiKey = f.readline()
CSE_ID = f.readline()
f.close()
os.environ["OPENAI_API_KEY"] = apiKey.strip()
search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY", googleApiKey.strip()),
    google_cse_id=os.getenv("GOOGLE_CSE_ID", CSE_ID.strip())
)

INTENT_PROMPT_TEMPLATE = os.path.join(CURRENT_DIR, "prompt_templates/parse_intent.txt")
INTENT_LIST_TXT = os.path.join(CURRENT_DIR, "prompt_templates/intent_list.txt")
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(CURRENT_DIR, "prompt_templates/default_response.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(CURRENT_DIR, "prompt_templates/search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(CURRENT_DIR, "prompt_templates/search_compress.txt")
SOLUTION_PROMPT_TEMPLATE = os.path.join(CURRENT_DIR, "prompt_templates/solution.txt")

CHROMA_PERSIST_DIR = os.path.join(CURRENT_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "kakao-api-bot"

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

llm = ChatOpenAI(temperature=0.7, max_tokens=1024, model="gpt-3.5-turbo")

parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)
solution_chain = create_chain(
    llm=llm,
    template_path=SOLUTION_PROMPT_TEMPLATE,
    output_key="output",
)
default_chain = create_chain(
    llm=llm,
    template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE,
    output_key="output"
)
search_value_check_chain = create_chain(
    llm=llm,
    template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
    output_key="output",
)
search_compression_chain = create_chain(
    llm=llm,
    template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
    output_key="output",
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    context["related_web_search_results"] = search_tool.run(user_message)

    has_value = search_value_check_chain.run(context)

    print(has_value)
    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
HISTORY_DIR = os.path.join(CURRENT_DIR, "chat_histories")

new_data = [
    {"type": "human", "data": {"content": "", "additional_kwargs": {}, "example": False}},
    {"type": "ai", "data": {"content": "", "additional_kwargs": {}, "example": False}}
]
json_file = os.path.join(HISTORY_DIR, "hist.json")
with open(json_file, 'w') as file:
    json.dump(new_data, file)

def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)

def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)

def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)

def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )
    return memory.buffer

def gernerate_answer(user_message, conversation_id: str='hist') -> dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    context["chat_history"] = get_chat_history(conversation_id)

    # intent = parse_intent_chain(context)["intent"]
    intent = parse_intent_chain.run(context)

    if intent == "social":
        context["related_documents"] = query_db(context["user_message"])
        answer = solution_chain.run(context)
    elif intent == "sync":
        context["related_documents"] = query_db(context["user_message"])
        answer = solution_chain.run(context)
    elif intent == "talkchannel":
        context["related_documents"] = query_db(context["user_message"])
        answer = solution_chain.run(context)
    else:
        context["related_documents"] = query_db(context["user_message"])
        context["compressed_web_search_results"] = query_web_search(
            context["user_message"]
        )
        answer = default_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}

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
        self.answer = gernerate_answer(self.text)["answer"]

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
            placeholder="KakaoAPIHelper에게 물어보세요!",
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
