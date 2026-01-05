import json
import os
from dotenv import load_dotenv

from transformers import pipeline
import openai

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Constants
# -----------------------------
TEMPERATURES = [0.2, 0.7, 1.2]
HF_MODEL_NAME = "distilgpt2"

# -----------------------------
# HuggingFace Generator
# -----------------------------
hf_generator = pipeline(
    "text-generation",
    model=HF_MODEL_NAME
)

def generate_hf_text(prompt):
    results = {}
    for temp in TEMPERATURES:
        output = hf_generator(
            prompt,
            max_length=120,
            temperature=temp,
            do_sample=True
        )[0]["generated_text"]

        results[f"temp={temp}"] = output
    return results

# -----------------------------
# OpenAI Generator
# -----------------------------
def generate_openai_text(prompt):
    results = {}
    for temp in TEMPERATURES:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        results[f"temp={temp}"] = response.choices[0].message["content"]
    return results

# -----------------------------
# Main
# -----------------------------
def main():
    user_input = input("Enter your prompt: ")

    hf_outputs = generate_hf_text(user_input)
    api_outputs = generate_openai_text(user_input)

    final_output = {
        "input": user_input,
        "outputs": {
            "huggingface": hf_outputs,
            "api_model": api_outputs
        }
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\n✅ Output saved to output.json")

if __name__ == "__main__":
    main()
