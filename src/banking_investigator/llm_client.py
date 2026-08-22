from groq import Groq

from banking_investigator.config.settings import settings


client = Groq(api_key=settings.llm_api_key)