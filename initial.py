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

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    query = request.query
    # Updated prompt template focused on detailed, step-by-step guidance
    prompt_template = f"""
        You will be given a query that is listed in the section where <Inputs> is.

        <Task>
        AI QUERY AGENT Directive
        </Task>

        <Inputs>
        <Query> {query} </Query>
        </Inputs>

        <Instructions>
        As an AI QUERY AGENT, your directive encompasses engaging in Query Analysis Expertise Protocol and initiating Attribute Determination Mastery Procedures, followed by activating the Audience-Tailored Summarization Sequence. This comprehensive task is aimed at providing in-depth query evaluation, attribute identification, and tailored summaries. Follow these steps:

        1. **In-Depth Query Analysis**:
           - Employ your full capabilities to analyze the provided meta-query inside the <Inputs> section. This involves understanding the query's content, functionality, and underlying context.

        2. **Query Attribute Identification**:
           - Determine the query's type, category, subcategory, topic, intent, and key information. Use analytical skills to identify these attributes accurately.

        3. **Functionality Summaries**:
           - Provide concise, optimized summaries tailored to different audience levels:
             - GPT prompt engineers: Offer technical insight into the query's structure and purpose.
             - Basic AI/GPT users: Present a simplified explanation of the query's functionality.
             - 15-year-olds: Use engaging and understandable language to explain the query's objective.

        4. **Comprehensive Data Field Population**:
           - Complete additional data fields covering aspects like query complexity, optimization, response characteristics, sentiment, readability, context preservation, entity extraction, coherence, bias detection, and confidence.
           - Further populate fields related to context, prompts, sentiment, domain research, sources, findings, concepts, GPT-4 capabilities, ambiguities, solutions, alternatives, integration, response generation, bottlenecks, post-processing, evaluation, and iteration.

        5. **Query Naming and Categorization**:
           - Create a concise, descriptive name for the query following the format: "Concise Title up to Twelve Words".
           - Determine the best location for the query within the provided folder structure hierarchy, maintaining logical consistency.

        **User Interaction Guidelines**:
        - If no query is provided initially, request the user to input a query.
        - Format your response in markdown to enhance readability.
        - Adhere strictly to the specified tasks and rules outlined above.

        **Iterative Refinement Cycle**:
        - After completing the query evaluation, review your analysis, attribute identification, summaries, and data field completion for accuracy and comprehensiveness.
        - Identify areas for improvement in understanding, categorization, and evaluation of the query.
        - Refine your approach based on insights gained and apply it to subsequent query evaluations. Repeat the cycle for continuous improvement.
        
        **Task Definition**:
        - Formulate the final task that encompasses all the insights derived from the query, clearly stating what needs to be accomplished next. This task should be actionable, specific, and directly linked to the insights and attributes identified during the query analysis.
        Ensure your work aligns with the AI QUERY AGENT Directive's objectives, following the specified guidelines and protocols.
        </Instructions>
        """
    parser = StrOutputParser()
    
    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template,
    )

    llm_chain = prompt | llm | parser
    
    result_chain = llm_chain.invoke({"query": query})

    query_payload = {"query": result_chain}
    render_url = "https://something"
    response = requests.post(render_url, json=query_payload)
    if response.status_code == 200:
        return {"message": "Query successfully sent to next part of the render chain process."}
    else:
        return {
            "error": "Failed to send render query.",
            "statusCode": response.status_code,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
