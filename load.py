from transformers import Trainer, TrainingArguments, BertTokenizer
from transformers import DataCollatorWithPadding
from peft import AutoPeftModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import evaluate


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


lora_model = AutoPeftModelForSequenceClassification.from_pretrained("bert-lora")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset = load_dataset("rotten_tomatoes")
test_dataset = dataset["test"].map(tokenize_function, batched=True)

# Evaluate the model
trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="output",
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        evaluation_strategy="epoch",
        metric_for_best_model="accuracy",
    ),
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

results = trainer.evaluate()
print(results)
