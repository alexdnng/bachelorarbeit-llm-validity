import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
load_dotenv()

print("LangChain funktioneirt")

#client = OpenAI(
#    api_key=os.getenv("OPENAI_API_KEY")
#)

#models = client.models.list()

#for model in models.data:
 #   print(model.id)