import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import (
    MAX_PROFILE_GENERATION_TOKENS,
    MAX_TWIN_RESPONSE_TOKENS,
    MAX_JUDGE_TOKENS,
)

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")


def generate_profile(prompt, temperature, top_p, model_name, max_tokens, provider):

    # -------------------------
    # OPENAI
    # -------------------------
    if provider == "openai":

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

    # -------------------------
    # OLLAMA
    # -------------------------
    elif provider == "ollama":

        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=top_p
        )
    elif provider == "gemini":
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            disable_streaming=True,
            thinking_budget=0,
        )

    else:
        raise ValueError(f"Invalid provider: {provider}")

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

    return generate_profile(
        prompt,
        temperature,
        top_p,
        model_name,
        MAX_PROFILE_GENERATION_TOKENS,
        "openai"
    )


def generate_twin_response(prompt, temperature, top_p, model_name):
    return generate_profile(
        prompt,
        temperature,
        top_p,
        model_name,
        MAX_TWIN_RESPONSE_TOKENS,
        MODEL_PROVIDER
    )


def generate_judge_response(prompt, temperature, top_p, model_name):
    return generate_profile(
        prompt,
        temperature,
        top_p,
        model_name,
        MAX_JUDGE_TOKENS,
        "openai"
    )