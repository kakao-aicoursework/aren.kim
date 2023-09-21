import pynecone as pc

class KakaoapichatbotConfig(pc.Config):
    pass

config = KakaoapichatbotConfig(
    app_name="kakaoAPIChatBot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)