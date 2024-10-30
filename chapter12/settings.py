import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str = ""
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "agent-book"

    # for Application
    # 2024/10/02から gpt-4o と指定すると gpt-4o-2024-08-06 を指します。2024/10/01以前は、 gpt-4o-2024-05-03を指していました。
    openai_smart_model: str = "gpt-4o-2024-08-06"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_smart_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.0
    default_reflection_db_path: str = "tmp/reflection_db.json"

    def __init__(self, **values) -> None:
        super().__init__(**values)
        self._set_env_variables()

    def _set_env_variables(self) -> None:
        for key in self.__annotations__.keys():
            if key.isupper():
                os.environ[key] = getattr(self, key)
