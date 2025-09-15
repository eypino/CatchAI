from datasets import Dataset

# Frases en español y sus glosas asociadas
data = {
    "es": [
        "Hola, cómo estás?",
        "Voy a la universidad",
        "Quiero comer pan",
        "El perro corre rápido",
        "Muchas gracias"
    ],
    "glosas": [
        "HOLA CÓMO ESTAR",
        "IR UNIVERSIDAD",
        "QUERER COMER PAN",
        "PERRO CORRER RÁPIDO",
        "MUCHO GRACIAS"
    ]
}

dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2)



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


max_length = 64

def preprocess(batch):
    inputs = tokenizer(batch["es"], max_length=max_length, truncation=True, padding="max_length")
    targets = tokenizer(batch["glosas"], max_length=max_length, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["es", "glosas"])



from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 4

args = Seq2SeqTrainingArguments(
    output_dir="./glosas_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    logging_dir="./logs",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

def traducir(texto):
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(traducir("Voy a la universidad"))  # → IR UNIVERSIDAD
print(traducir("Muchas gracias"))        # → MUCHO GRACIAS