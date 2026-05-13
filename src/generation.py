import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def generate_profile(prompt, temperature, top_p, model_name):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=300,
        top_p=1
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content


def generate_behavior_from_traits(traits, temperature, top_p, model_name="gpt-4.1-mini"):
    prompt = f"""
You are given a person's Big Five personality traits.

Extraversion: {traits['Extraversion']}
Agreeableness: {traits['Agreeableness']}
Conscientiousness: {traits['Conscientiousness']}
Neuroticism: {traits['Neuroticism']}
Openness: {traits['Openness']}

Write a realistic and detailed description of this person's behavior in daily life.
Focus on observable actions, habits, and social interactions.
Do NOT mention numbers or trait labels.
"""

    return generate_profile(prompt, temperature, top_p, model_name)