from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    query: str
    topic: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    query = request.query
    topic = request.topic
    # Updated prompt template focused on detailed, step-by-step guidance
    prompt_template = """
   Answer the following question: {query}
   About the following topic: {topic}
    """

    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["query", "topic"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"query": query, "topic": topic})

    return result_chain

