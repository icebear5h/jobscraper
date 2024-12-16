import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from openai import OpenAI
from toolhouse import Toolhouse
from groq import Groq

load_dotenv()

oclient = OpenAI(api_key=os.environ.get('OPEN_API_KEY'))
qclient = Groq(api_key=os.environ.get('GROQ_API_KEY'))
MODEL = "llama3-groq-70b-8192-tool-use-preview"


th = Toolhouse()

urlmessages = [{
    "role": "user",
    "content":
        """"
        Search the internet exahustively for summer internships that open in 2025 relating to ml, llm, ai, retrieval augmented generation etc, get as many results as possible.
        for the websearch tool, set the parameter for Number of results like something ridiculoulsy high like 35k, just return an easily parisible list of urls
        """
}]



def check_job_with_llm(title, webscrape):
    prompt = f"""
    You need to determine if this job listing follows the requirements given

    Criteria:
    - Must be an internship.
    - Must be for Summer 2025.
    - Must involve AI, Machine Learning, LLMs (Large Language Models), or RAG (Retrieval-Augmented Generation).
    - Must be open to bachelor's graduates (no advanced degree strictly required).

    Given this title and description:
    TITLE: {title}
    Web Scraped Result: {webscrape}

    Please respond with "YES" if it meets all criteria, or "NO" otherwise.
    """

    completion = oclient.chat.completions.create(
        model="o1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. You have been given a job title and a webpage layout."},
            {"role": "user", "content": prompt}
        ]
    )

    verdict = completion.choices[0]
    return verdict == "YES"

response = qclient.chat.completions.create(
  model=MODEL,
  messages=urlmessages,
  # Passes Code Execution as a tool
  tools=th.get_tools(),
)

# Runs the Code Execution tool, gets the result, 
# and appends it to the context
tool_run = th.run_tools(response)
urlmessages.extend(tool_run)

response = qclient.chat.completions.create(
  model=MODEL,
  messages=urlmessages,
  tools=th.get_tools(),
)

print(response.choices[0].message.content)

