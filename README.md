## Intro
This is a project for the generative AI udacity nanodegree program. <br>
The goal of this project is to apply lightweight fine-tuning to a foundation model. <br>

- Load a pre-trained model and evaluate its performance.
  - [x] Load a pretrained HF model
  - [x] Load and preprocess a dataset
  - [x] Evaluate the model on the dataset
- Perform PEFT using the pre-trained model.
  - [x] Create a PEFT model
  - [x] Train the PEFT model
  - [x] Save the PEFT model
- Perform inference using the fine-tuned model and compare the results with the pre-trained model.
  - [x] Load the PEFT model
  - [x] Evaluate the PEFT model on the dataset

## How to evaluate a pre-trained model?
The foundation model usually has two parts:
- Body
- Head

Head is the top layer that can be changed depending on the task. For example: text generation, classification, <br>
translation, Q&A, etc. <br>

## Notes
### LoraConfig
`r` is a parameter that sets the rank of the low-rank matrices used in LoRA adaptation. <br>
`r = 32` means low-rank matrices will have a dimensionality of 32. <br>
This directly affect the number of parameters in the model. <br>

`lora_alpha` specifies the scaling factor for the LoRA adaptation. <br>
It controls how much the adapted parameters(low-rank matrices) influence the original
weights of the model. `lora_alpha = 4` indicates a moderate influence, adjusting how much <br>
the LoRA modifications impact the model's behavior. <br>

`target_modules` lists the components of the Transformer model that will be adapted <br>
using LoRA. Typically, these are parts of the attention mechanism in Transformers, <br>
namely the `query`, `key`, and `value` projections. This means LoRA will specifically <br>
adapt these parts to better suit the new task without retraining them entirely. <br>

`lora_dropout` sets the dropout rate to be used in the LoRA layers. <br>
This helps in preventing overfitting by introducing some noise into the training process. <br>

`bias` determines the use of bias in the LoRA layers. <br>

`task_type` specified the type of task the model is being adapted for. <br>
`TaskType.SEQ_CLS` is used for sequence classification tasks. <br>
This field is a must for the configuration. <br>

`inference_mode` indicates whether the model is being configured for training or inference. <br>
Setting it to `False` means the model is being set up for training, allowing all training <br>
specific elements like dropout and other regularizations to be active. <br>

## Results
After applying LoRA on the BERT, the accuracy improved from 70% to 83% on the test dataset
```bash

BERT with LoRA:
{'eval_loss': 0.3936154544353485, 'eval_accuracy': 0.8358348968105066, 'eval_runtime': 12.0454, 'eval_samples_per_second': 88.499, 'eval_steps_per_second': 1.411}
```