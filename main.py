from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model

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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
    }


# Convert to a PEFT model
config = LoraConfig()
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="output",
        learning_rate=2e-3,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        do_train=True,
        do_eval=True,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        eval_steps=100,
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
