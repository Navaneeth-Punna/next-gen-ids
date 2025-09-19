# Install all required libraries in one go
!pip install -q torch transformers datasets peft bitsandbytes scikit-learn

# --- 1. Data Preparation ---
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from accelerate import Accelerator
import os
import torch
from peft import LoraConfig, get_peft_model

print("--- Step 1: Downloading and Preprocessing Data ---")

# Load the 'emotion' dataset directly from Hugging Face
raw_dataset = load_dataset("emotion")
df = pd.concat([raw_dataset['train'].to_pandas(), raw_dataset['validation'].to_pandas()])
print("Dataset downloaded and loaded successfully.")

# Convert to instruction-based format
def create_instruction_data(df):
    data = []
    for _, row in df.iterrows():
        log_content = row['text']
        label = row['label']
        prompt_template = (
            "### Instruction:\nAnalyze the following log and determine its sentiment. "
            "Classify it as one of the following: sadness, joy, love, anger, fear, or surprise. "
            "Explain why and provide an example of a similar log.\n\n"
            "### Log Entry:\n{log_content}\n\n"
            "### Output:"
        )
        label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
        output = f"Label: {label_map[label]}\nExplanation: The message expresses {label_map[label]}."
        data.append({"text": f"{prompt_template.format(log_content=log_content)}{output}"})
    return Dataset.from_pandas(pd.DataFrame(data))

instruction_dataset = create_instruction_data(df)
print("Instruction dataset created.")

# Split the dataset before tokenization
train_dataset, eval_dataset = instruction_dataset.train_test_split(test_size=0.1).values()
print("Data split into training and evaluation sets.")


# --- 2. LLM Fine-tuning ---


print("--- Step 2: Fine-tuning the LLM ---")

# Hugging Face login
# Please generate a new Hugging Face token with access to public gated repositories and paste it when prompted below.
# Go to: https://huggingface.co/settings/tokens to create a new token.
!huggingface-cli login
print("Hugging Face login command executed. Please check the output above for login status and ensure you used a token with appropriate permissions.")


# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load Model and Tokenizer
model_id = "EleutherAI/Pythia-160m" # Corrected model_id for Pythia
tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="<|endoftext|>")
# Set pad_token_id explicitly for clarity
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Disable gradient checkpointing for debugging
# model.gradient_checkpointing_enable()


# Tokenize the datasets
def tokenize_function(examples):
    tokenized_output = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256) # Reduced max_length
    return tokenized_output

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Remove the original 'text' column after tokenization
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])

# Set dataset format to torch
train_dataset.set_format("torch")
eval_dataset.set_format("torch")


print("Data preprocessed and ready for fine-tuning.")

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"], # Adjusted target modules for Pythia
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("Model loaded and prepared for fine-tuning.")

# Print requires_grad status of model parameters
print("Requires grad status of model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")


# Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, # Reduced eval batch size
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete. The trained model is ready for inference.")

# --- Evaluating the fine-tuned model ---
print("--- Evaluating the fine-tuned model ---")
evaluation_results = trainer.evaluate()
print(evaluation_results)   
