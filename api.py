from fastapi import FastAPI
from pydantic import BaseModel
from agent_api import sendPromptToAgent
from dotenv import load_dotenv
import os

load_dotenv("./settings.env") 

app = FastAPI()

@app.get('/')
def main():
   return {'response': 'main'}

# نستخدم BaseModel للحصول على prompt من body
class PromptBody(BaseModel):
    prompt: str

@app.post('/news')
def post_prompt(data: PromptBody):
    print("🔑 OPENROUTER_API=", str(os.getenv("OPENROUTER_API")))
    result =   sendPromptToAgent(data.prompt)
    
    return {'response': result}
