import os
import getpass
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

class Message(BaseModel):
    message: str

prompt = PromptTemplate.from_template(
    "User said: {msg}. Reply politely."
)

chain = prompt | llm

def chat(m: Message):
    response = chain.invoke({"msg": m.message})
    reply_text = response.content if hasattr(response, "content") else str(response)
    return {"reply": reply_text}

print(chat(Message(message="Hello, how are you?")))