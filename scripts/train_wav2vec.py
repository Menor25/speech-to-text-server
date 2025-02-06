import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Load custom dataset
dataset = load_dataset("json", data_files={"train": "../data/train.json", "test": "../data/test.json"})

# Preprocess dataset
def preprocess_data(sample):
    input_values = processor(sample["audio"], sampling_rate=16000).input_values[0]
    labels = processor.tokenizer(sample["text"]).input_ids
    return {"input_values": input_values, "labels": labels}

dataset = dataset.map(preprocess_data)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/wav2vec2",
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    learning_rate=3e-4,
    num_train_epochs=3,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
trainer.train()

# Save fine-tuned model
model.save_pretrained("../models/wav2vec2")
processor.save_pretrained("../models/wav2vec2")
