from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from peft import AutoPeftModelForSequenceClassification
from accelerate import dispatch_model, infer_auto_device_map
import numpy as np
import evaluate
import torch
import argparse


# Check the environment and select device
def check_environment():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            torch.cuda.set_device(0)
            print(f"Multiple CUDA devices found, deviceNum: {num_devices}. Setting default device to GPU 0.")
        else:
            print("Single CUDA device found. Using GPU 0 as default device.")
    else:
        print("CUDA is not available. Using CPU.")
    return


# Dispatch the model to the device
def auto_dispatch_model(model):
    device_map = infer_auto_device_map(model)
    return dispatch_model(model, device_map=device_map)


def train_model(model, train_dataset, vali_dataset, tokenizer, compute_metrics, save_path=None):
    trainer = Trainer(
        model=model,
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
    if save_path is not None:
        trainer.model.save_pretrained(save_path)
    return


def evaluate_mode(model, test_dataset, tokenizer, compute_metrics):
    # Evaluate the model
    trainer = Trainer(
        model=model,
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
    return


def main(args):
    base_model = True

    if args.model == 'base':
        print("Use base model...")
    elif args.model == 'lora':
        print("Use LoRA model...")
        base_model = False
    else:
        print("Invalid model argument. Please specify 'base' or 'lora'.")
        return

    check_environment()

    # Load pre-trained HF model BERT, and add a classification head to it so that
    # we can evaluate the model on the Rotten Tomatoes dataset.
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = 2
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    # Freeze all the parameters of the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("rotten_tomatoes")
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    vali_dataset = dataset["validation"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Train the BERT base model
    if base_model:
        dispatched_model = auto_dispatch_model(model)
        train_model(dispatched_model, train_dataset, vali_dataset, tokenizer, compute_metrics)
        evaluate_mode(dispatched_model, test_dataset, tokenizer, compute_metrics)
        return

    # Use LoRA to fine-tune the model
    config = LoraConfig(
        r=32,
        lora_alpha=4,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False
    )

    saved_model_path = "bert-lora"
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    lora_model = auto_dispatch_model(lora_model)
    train_model(lora_model, train_dataset, vali_dataset, tokenizer, compute_metrics,
                save_path=saved_model_path)

    # Load the LoRA model
    lora_model = AutoPeftModelForSequenceClassification.from_pretrained(saved_model_path)
    lora_model = auto_dispatch_model(lora_model)
    evaluate_mode(lora_model, test_dataset, tokenizer, compute_metrics)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main function with base and lora arguments")
    parser.add_argument('model', choices=['base', 'lora'], help="Specify the model type: 'base' or 'lora'")
    main_args = parser.parse_args()
    main(main_args)
