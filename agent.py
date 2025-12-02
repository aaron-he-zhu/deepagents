"""Simple deep agent for testing."""

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

# Use Google Gemini model
model = init_chat_model("google_genai:gemini-2.0-flash")

# Create a simple deep agent with Google Gemini model
agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful AI assistant. Be concise and helpful in your responses.",
)

