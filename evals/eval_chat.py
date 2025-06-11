from braintrust import Eval, projects, wrap_openai, init_dataset
from autoevals import Faithfulness
from openai import OpenAI
from dotenv import load_dotenv
import os
from helpers.helpers import chat

load_dotenv()

client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
)
PROJECT_NAME=os.getenv("BRAINTRUST_PROJECT_NAME")
project = projects.create(name=PROJECT_NAME)

def faithfulness(input, output):
    return Faithfulness()(input=input, output=output["output"], context=output["context"])

Eval(
    name=PROJECT_NAME,
    task=chat,
    data=init_dataset(project=PROJECT_NAME, name="QandAEvalCases"),
    scores=[faithfulness],
    max_concurrency=5
)




