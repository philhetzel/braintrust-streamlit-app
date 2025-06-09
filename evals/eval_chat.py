from braintrust import load_prompt, Eval, projects, wrap_openai, init_dataset, init_function
from openai import OpenAI
from dotenv import load_dotenv
import os
from tools.retriever import tool_definition
from helpers.helpers import chat
from scorers.scores import forgetfulness
import asyncio

load_dotenv()

client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
)
PROJECT_NAME=os.getenv("BRAINTRUST_PROJECT_NAME")
project = projects.create(name=PROJECT_NAME)

Eval(
    name=PROJECT_NAME,
    task=chat,
    data=init_dataset(project=PROJECT_NAME, name="chat-eval"),
    scores=[forgetfulness],
    max_concurrency=5
)




