import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
import os

from playsound import playsound

import sounddevice as sd
print(sd.query_devices())


# -------------------------------
# Load your trained model
# -------------------------------
model_path = "./RR2/corellm"  # path to your saved model
#model_path = "./final/distilgpt2-food-ai"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=120,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

# -------------------------------
# Speech-to-text function
# -------------------------------
recognizer = sr.Recognizer()


def speech_to_text(duration=6):
    mic_index = 9  # Replace with your actual Razer Seiren index
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"🎤 Recording for {duration} seconds...")
        audio = recognizer.record(source, duration=duration)  # fixed duration recording

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"⚠️ STT request failed: {e}")
        return ""



# -------------------------------
# Text-to-speech function
# -------------------------------
def speak(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts_file = "response.mp3"
    tts.save(tts_file)
    playsound(tts_file)
    os.remove(tts_file)

# -------------------------------
# Generate answer from model
# -------------------------------
def generate_answer(question):
    prompt = f"Q: {question}\nA:"
    response = qa_pipeline(prompt, num_return_sequences=1)
    text = response[0]["generated_text"]
    # Only return the answer part
    return text.split("A:")[-1].strip()
    #answer = text.split("A:")[-1].strip()
    # Stop at the first full stop
    #if "." in answer:
        #answer = answer.split(".")[0] + "."
    #return answer

# -------------------------------
# Chatbot loop
# -------------------------------
def chatbot_loop():
    print("🎙️ Chatbot ready (type 'q' to quit)")
    while True:
        command = input("\nPress ENTER to speak or 'q' to quit: ").strip().lower()
        if command == "q":
            print("👋 Exiting")
            break

        question = speech_to_text(duration=6)
        if not question:
            continue

        print("🗣️ You:", question)
        answer = generate_answer(question)
        print("🤖:", answer)
        speak(answer)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    chatbot_loop()

