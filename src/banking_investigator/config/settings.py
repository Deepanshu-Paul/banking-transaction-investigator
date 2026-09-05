from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str

    db_statement_timeout_ms: int = 2000

    llm_provider: str
    llm_model: str

    groq_api_key: str | None = None
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is required when LLM_PROVIDER=groq"
                )
            return self.groq_api_key

        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
                )
            return self.openai_api_key

        raise ValueError(
            f"Unsupported LLM_PROVIDER: {self.llm_provider}"
        )

    @property
    def postgres_conn_string(self) -> str:
        return self.database_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )


settings = Settings()