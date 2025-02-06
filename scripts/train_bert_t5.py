from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load dataset
dataset = load_dataset("json", data_files={"train": "../data/train.json", "test": "../data/test.json"})

# Preprocess text data
def preprocess_data(sample):
    inputs = f"fix: {sample['text']}"
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    return {"input_ids": model_inputs["input_ids"][0], "labels": model_inputs["input_ids"][0]}

dataset = dataset.map(preprocess_data)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/bert_t5",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
trainer.train()

# Save fine-tuned model
model.save_pretrained("../models/bert_t5")
tokenizer.save_pretrained("../models/bert_t5")
