# Augmenting Knowledge Data

from kaggle.api.kaggle_api_extended import KaggleApi
import os

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""

api = KaggleApi()
api.authenticate()

api.dataset_download_files("farahmostafaahmed/benefits-of-fruits-and-vegetables", path="datasets/food", unzip=True)
api.dataset_download_files("yapwh1208/chatbot-ai-q-and-a", path="datasets/ai", unzip=True)
api.dataset_download_files("utsavdey1410/food-nutrition-dataset", path="datasets/vit", unzip=True)

import pandas as pd

# Food Benefits dataset
food_df = pd.read_csv("datasets/food/fruit_vegetable_benefits.csv")
print(food_df.head())

# AI QnA dataset
ai_df = pd.read_csv("datasets/ai/AI.csv")
print(ai_df.head())

# Food Vitamin Dataset
vit_df = pd.read_csv("datasets/vit/FINAL FOOD DATASET/FOOD-DATA-GROUP3.csv",nrows=275)
print(vit_df.columns)

import pandas as pd

food_qa = []

# Dataset to QnA
for _, row in food_df.iterrows():
    name = row['Name']
    type_ = row['Type']
    benefit = row['Benefit']

    templates = [
        f"What are the health benefits of {name}?",
        f"Why is {name} good for you?",
        f"How does {name} help maintain good health?",
        f"Can you tell me the benefits of eating {name}?",
        f"What makes {name} a healthy {type_}?",
        f"How can {name} support overall wellness?"
    ]

    for q in templates:
        a_templates = [
            f"{name} is a {type_} that {benefit}.",
            f"Health-wise, {name} provides {benefit}.",
            f"One benefit of {name} ({type_}) is that it {benefit}.",
            f"Consuming {name}, a {type_}, can help {benefit}.",
            f"{name} ({type_}) is known to {benefit}.",
        ]
        
        for a in a_templates:
            food_qa.append(f"Q: {q}\nA: {a}")

ai_qa = []

for _, row in ai_df.iterrows():
    
    answer = row['Answer'].strip().replace("\n", " ")
    question = row['Question'].strip()
    ai_qa.append(f"Q: {question}\nA: {answer}")

NUTRIENT_BENEFITS = {
    "Protein": "supports muscle repair and growth",
    "Dietary Fiber": "supports digestion and gut health",
    "Water": "helps with hydration",
    "Vitamin A": "supports vision and immune function",
    "Vitamin B3": "supports nervous system health",
    "Vitamin B5": "supports hormone and energy production",
    "Vitamin C": "supports immune health and collagen production",
    "Vitamin D": "supports calcium absorption and bone health",
    "Vitamin E": "acts as an antioxidant protecting cells",
    "Calcium": "supports strong bones and teeth",
    "Iron": "supports oxygen transport in the blood",
    "Magnesium": "supports muscle and nerve function"
}

THRESHOLDS = {
    "Water": {"high": 70, "moderate": 50},
    "Protein": {"high": 10, "moderate": 5},
    "Dietary Fiber": {"high": 5, "moderate": 2.5},
    "Vitamin A": {"high": 20, "moderate": 10},
    "Vitamin C": {"high": 20, "moderate": 10},
    "Vitamin B3": {"high": 15, "moderate": 5},
    "Vitamin B5": {"high": 15, "moderate": 5},
    "Vitamin D": {"high": 20, "moderate": 10},
    "Vitamin E": {"high": 15, "moderate": 5},
    "Calcium": {"high": 20, "moderate": 10},
    "Iron": {"high": 20, "moderate": 10},
    "Magnesium": {"high": 20, "moderate": 10},
}

def classify_nutrient(value, nutrient):
    if nutrient not in THRESHOLDS:
        return None

    if value >= THRESHOLDS[nutrient]["high"]:
        return "high"
    elif value >= THRESHOLDS[nutrient]["moderate"]:
        return "moderate"
    else:
        return "low"

def nutrient_answer(food, nutrient, level, benefit):
    if level == "high":
        return f"{food} is high in {nutrient.lower()}, which {benefit}."
    elif level == "moderate":
        return f"{food} contains a moderate amount of {nutrient.lower()}, which {benefit}."
    else:
        return f"{food} is not a significant source of {nutrient.lower()}."

vit_qa = []

for _, row in vit_df.iterrows():
    food = row["food"].title()

    for nutrient, benefit in NUTRIENT_BENEFITS.items():
        if nutrient not in row:
            continue

        level = classify_nutrient(row[nutrient], nutrient)

        if not level:
            continue

        if level == "low":
            vit_qa.append(
                f"Q: Is {food} a good source of {nutrient}?\n"
                f"A: {food} is not a significant source of {nutrient.lower()}."
            )
            continue

        vit_qa.append(
            f"Q: What does {nutrient} in {food} do for the body?\n"
            f"A: {nutrient_answer(food, nutrient, level, benefit)}"
        )

        vit_qa.append(
            f"Q: Is {food} rich in {nutrient}?\n"
            f"A: {nutrient_answer(food, nutrient, level, benefit)}"
        )

from random import shuffle
shuffle(vit_qa)
vit_qa = vit_qa[:1000]

knowledge_texts = food_qa + ai_qa + vit_qa
shuffle(knowledge_texts)
print(knowledge_texts[:5])

# Augmenting Style Dataset
import wikipedia
import re

pages = [
    "Pineapple",
    "Potato",
    "Blackberry",
    "Avocado",
    "Watermelon",
    "Cucumber",
    "Vitamin",
    "Vitamin_A",
    "Artificial_intelligence",
    "Deep_learning",
    "Convolutional_neural_network",
    "Ethics_of_artificial_intelligence",
    "Human–robot_interaction"
]


style_texts = []
for title in pages:
    try:
        page = wikipedia.page(title)
        paragraphs = page.content.split("\n\n")[:3]
        for p in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            style_texts.extend([s for s in sentences if len(s.split()) > 12])
    except:
        pass

style_texts = [
    p.replace("==", "").strip()
    for p in style_texts
    if len(p.split()) > 20
]
shuffle(style_texts)
print(style_texts)
print(len(style_texts))

# Training 

import torch
import math
import wandb
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from random import shuffle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from codecarbon import EmissionsTracker
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=96):
        self.examples = []
        for t in texts:
            enc = tokenizer(
                t,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            )
            self.examples.append(enc)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze()
        }
smoothie = SmoothingFunction().method4

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits).view(-1, logits.shape[-1]),
        torch.tensor(labels).view(-1),
        ignore_index=tokenizer.pad_token_id
    )

    perplexity = math.exp(loss.item())

    # Simple BLEU-2 proxy (not perfect, but acceptable for coursework)
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    bleu_scores = []

    for pred, label in zip(preds, labels):
        pred_tokens = tokenizer.decode(pred, skip_special_tokens=True).split()
        label_tokens = tokenizer.decode(label, skip_special_tokens=True).split()
        if len(pred_tokens) > 1 and len(label_tokens) > 1:
            bleu = sentence_bleu(
                [label_tokens],
                pred_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoothie
            )
            bleu_scores.append(bleu)

    bleu2 = sum(bleu_scores) / max(1, len(bleu_scores))

    return {
        "perplexity": perplexity,
        "bleu2": bleu2
    }
class TextGenerationCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:
            prompt = "Q: What are the health benefits of carrots?\nA:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=2.0
            )

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            wandb.log({"generated_text": text})

knowledge_dataset = TextDataset(knowledge_texts, tokenizer)

knowledge_args = TrainingArguments(
    output_dir="./distilgpt2-knowledge",
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    report_to="wandb",
    run_name="distilgpt2-knowledge"
)

tracker = EmissionsTracker(project_name="distilgpt2-knowledge")
tracker.start()

knowledge_trainer = Trainer(
    model=model,
    args=knowledge_args,
    train_dataset=knowledge_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics,
    callbacks=[TextGenerationCallback()]
)

print("=== Stage 1: Knowledge Fine-tuning ===")
knowledge_trainer.train()
tracker.stop()

model.save_pretrained("./distilgpt2-knowledge")
tokenizer.save_pretrained("./distilgpt2-knowledge")
torch.cuda.empty_cache()


style_dataset = TextDataset(style_texts, tokenizer, max_length=256)

style_args = TrainingArguments(
    output_dir="./distilgpt2-style",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=500,
    report_to="wandb",
    run_name="distilgpt2-style"
)

tracker = EmissionsTracker(project_name="distilgpt2-style")
tracker.start()

style_trainer = Trainer(
    model=model,
    args=style_args,
    train_dataset=style_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[TextGenerationCallback()]
)

print("=== Stage 2: Style Fine-tuning ===")
style_trainer.train()
tracker.stop()

model.save_pretrained("./distilgpt2-food-ai")
tokenizer.save_pretrained("./distilgpt2-food-ai")

# Test Code
# -------------------------------
# Example generation
# -------------------------------
from transformers import pipeline

qa_pipeline = pipeline(
    "text-generation",
    model="./distilgpt2-food-ai",
    tokenizer="./distilgpt2-food-ai",
    max_length=100,
    do_sample=True,
    top_p=0.9,
    top_k=50,
    temperature=0.7
)

# Example test questions
test_questions = [
    "What are the health benefits of carrots?",
    "Why is spinach good for the body?",
    "How does broccoli support bone health?",
    "Is sweet potato good for digestion?",
    "How do CNNs work in image recognition?",
    "Why is deep learning useful for AI tasks?"
]

for i, question in enumerate(test_questions, 1):
    prompt = f"Q: {question}\nA:"
    response = qa_pipeline(prompt, num_return_sequences=1)
    answer = response[0]['generated_text'].split("A:")[-1].strip()
    print(f"{i}. Q: {question}")
    print(f"   A: {answer}\n")
