from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import dispatch_model, infer_auto_device_map
import numpy as np
import evaluate
import torch

# Check the environment
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        torch.cuda.set_device(0)
        print(f"Multiple CUDA devices found, deviceNum: {num_devices}. Setting default device to GPU 0.")
    else:
        print("Single CUDA device found. Using GPU 0 as default device.")
else:
    print("CUDA is not available. Using CPU.")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.num_labels = 2

# Load datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset["train"].map(tokenize_function, batched=True)
vali_dataset = dataset["validation"].map(tokenize_function, batched=True)
test_dataset = dataset["test"].map(tokenize_function, batched=True)

# Add a classification head
model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=config
)

# Freeze all the parameters of the base model
for param in model.base_model.parameters():
    param.requires_grad = False


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Convert to a PEFT model
config = LoraConfig(
        r=32,
        lora_alpha=4,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False
        )

lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()
# For multi-GPU environment
device_map = infer_auto_device_map(lora_model)
lora_model = dispatch_model(lora_model, device_map=device_map)

trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="output",
        learning_rate=5e-5,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        do_train=True,
        do_eval=True,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    ),
    train_dataset=train_dataset,
    eval_dataset=vali_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.model.save_pretrained("bert-lora")
