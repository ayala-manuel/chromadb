from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv("resources/llm/.env")
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set. Check resources/llm/.env")

client = OpenAI(api_key=API_KEY)

def basic_rag_query(user_query: str, results : list, prompt_type : str) -> str:
    """
    Perform a basic query to the OpenAI API.
    Args:
        user_query (str): The user's query to be sent to the OpenAI API.
        results (list): A list of results to be included in the query context.
        prompt_type (str): The type of prompt to use for the query.
    Returns:
        str: The response from the OpenAI API.
    """

    with open (f"resources/prompts/{prompt_type}.txt", "r") as file:
        basic_prompt = file.read()
        clean_response = results["response"]
        documents = clean_response["documents"]
        titles = ".\n".join([item["title"] for item in clean_response["metadatas"][0]])
        dates = ".\n".join([item["date"] for item in clean_response["metadatas"][0]])
        retrieved_text = f"""
              Documents: {documents} \n
            ------------------------------\n
                Titles: {titles} \n
            --------------------------\n
                Dates: {dates}
              """

    system_prompt = basic_prompt.format(
        RESULTS=retrieved_text,
        QUERY=user_query
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt}
        ],
        max_tokens=1000
    )
    
    if response.choices:
        return response.choices[0].message.content.strip()
    else:
        return "No response from the model."