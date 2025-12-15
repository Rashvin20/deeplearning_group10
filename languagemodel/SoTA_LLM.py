import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI()

def sota_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content

import evaluate
bleu = evaluate.load("sacrebleu")

preds, refs = [], []

for sample in test_pairs[:50]:  # limit for API cost
    pred = sota_response(sample["prompt"])
    preds.append(pred)
    refs.append([sample["response"]])

bleu_score = bleu.compute(predictions=preds, references=refs)
print("SOTA BLEU-2:", bleu_score["precisions"][1])
