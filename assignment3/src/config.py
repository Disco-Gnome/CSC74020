# Define some dummy data
MODELS = ["OpenAI: gpt-3.5-turbo", "OpenAI: gpt-3.5-turbo-16k", "OpenAI: gpt-4", "OpenAI: gpt-4-1106-preview",
          "Anthropic: claude-3-opus-20240229","Google: gemini-pro"]

#the code will only use the first model listed currently
EMBEDDING_MODELS = ["all-MiniLM-L6-v2"]

TEMPERATURE = .5

#max tokens for openai llm or other llm used in the app. you will add if condition to check if context is too long
MAX_TOKENS = 8193

APP_NAME = "I'm Lucy - helping you chat with your PDF"

# make sure to include the trailing slash
PROCESSED_DOCUMENTS_DIR = "../data/processed/"
REPORTS_DOCUMENTS_DIR = "../data/reports/"
