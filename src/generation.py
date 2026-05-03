
from dotenv import load_dotenv
load_dotenv()


import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_profile(prompt, temperature, top_p):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=300
    )

    return response.choices[0].message.content

def generate_behavior_from_traits(traits, temperature, top_p):
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

    return generate_profile(prompt, temperature, top_p)