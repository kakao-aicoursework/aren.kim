import pynecone as pc

class KakaosyncConfig(pc.Config):
    pass

config = KakaosyncConfig(
    app_name="kakaoSync",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)