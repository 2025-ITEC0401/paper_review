from transformers import AutoTokenizer

model_path = './models/llama-3.1-8b-instruct'

try:
    print(f"Attempting to load tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("---" * 20)
    print("✅ Tokenizer loaded successfully!")
    print(tokenizer)
    print("---" * 20)
except Exception as e:
    print("---" * 20)
    print(f"❌ Failed to load tokenizer.")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
    print("---" * 20)

