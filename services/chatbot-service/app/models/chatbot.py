from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv

# Load api key from env file
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_TOKEN")

if not HF_ACCESS_TOKEN:
    raise ValueError("HF_ACCESS_TOKEN not found")

# Get model
MODEL_NAME = "AIForEdu/Llama-2-7b-weather-chatbot"

try:
    print(f"Attempting to load model: {MODEL_NAME}...")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Successfully loaded model.")
    
except Exception as e:
    raise ValueError(
        f"Failed to load model or tokenizer for {MODEL_NAME}. "
        f"Original error: {e}"
    )