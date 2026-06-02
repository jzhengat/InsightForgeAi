from openai import OpenAI
import os

# uses environment variable OPENAI_API_KEY automatically
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_answer(context, question):
    prompt = f"""
You are a Business Intelligence AI Assistant.

Use the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Give insights, trends, and recommendations.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content