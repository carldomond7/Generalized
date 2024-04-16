from fastapi import FastAPI, HTTPException
import uvicorn
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/accept/")
async def process_request(request: Request):
    try:
        # Get the JSON payload from the request
        data = await request.json()

        # Log the received data
        print(f"Received data: {data}")

        # Extract the 'query' field from the nested JSON structure
        query = data.get("message", {}).get("functionCall", {}).get("parameters", {}).get("query")

        if query is None:
            raise HTTPException(status_code=400, detail="Missing 'query' field in the request payload")

        llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

        prompt_template = """
        Answer the following question: {query}
        Ensure that your response is grammatically correct in every way shape form and fashion. I am referring to the text of your response, it must be absolutely flawless grammar no spelling mistakes at all. Write all numbers in letter form,
        for example If you are given a number such as "1" you are to write it as "one"
        """

        # Define the prompt structure
        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_template,
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Pass the context and question to the Langchain chain
        result_chain = llm_chain.invoke({"query": query})

        return result_chain

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
