import wandb
import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoConfig
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, return_tensors='pt', padding="max_length")

dataset = load_dataset("neurae/dnd_style_intents", use_auth_token="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")

train_dataset = dataset["train"]
eval_dataset = dataset["eval"]
test_dataset = dataset["test"]

train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
eval_tokenized_dataset = eval_dataset.map(preprocess_function, batched=True)
test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True)

train_tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
eval_tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
test_tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metrics = dict()
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = np.argmax(logits, axis=-1)
    metrics = {
        'accuracy': accuracy.compute(predictions=preds, references=labels),
        'precision-micro': precision.compute(predictions=preds, references=labels, average='micro'),
        'precision-macro': precision.compute(predictions=preds, references=labels, average='macro'),
        'recall-micro': recall.compute(predictions=preds, references=labels, average='micro'),
        'recall-macro': recall.compute(predictions=preds, references=labels, average='macro'),
        'f1-micro': f1.compute(predictions=preds, references=labels, average='micro'),
        'f1-macro': f1.compute(predictions=preds, references=labels, average='macro')
    }
    return metrics

label2id = {
    "Attack":0,
    "Complete quest": 1,
    "Deliver":2,
    "Drival":3,
    "Exchange": 4,
    "Farewell": 5,
    "Follow": 6,
    "General": 7,
    "Greeting": 8,
    "Join": 9,
    "Joke": 10,
    "Knowledge":11,
    "Move":12,
    "Protect": 13,
    "Recieve quest": 14,
    "Message": 15,
    "Threat": 16
}

id2label = {value:key for key, value in label2id.items()}

config = AutoConfig.from_pretrained("bert-base-cased", label2id=label2id, id2label=id2label)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", config=config)

wandb.init(project="nlu embeddings", name="bert")

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    output_dir='./models/bert',
    overwrite_output_dir=True,
    logging_strategy='steps',
    logging_dir='./models/bert/logs',
    logging_steps=100,
    learning_rate=0.00013253246006813307,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.0695551706838144,
    report_to='wandb',
    load_best_model_at_end=True,
    save_strategy='epoch',
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    lr_scheduler_type="constant",
    torch_compile=True,
    bf16=True,
    optim="adafactor",
    gradient_accumulation_steps=4
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()
_, _, metrics = trainer.predict(test_dataset=test_tokenized_dataset)
print(metrics)
wandb.finish(exit_code=0)
