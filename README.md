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

```bash
usage: main.py [-h] {base,lora}

Main function with base and lora arguments

positional arguments:
  {base,lora}  Specify the model type: 'base' or 'lora'
```

## Question during the project
Q: How to evaluate a pre-trained model?
A: The foundation model usually has two parts:
1. Body
2. Head

Head is the top layer that can be changed depending on the task. For example: text generation, classification, <br>
translation, Q&A, etc. So the evaluation is done by adding a head and training it on the task-specific dataset. <br>

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
python3 main.py base
{'eval_loss': 0.6603750586509705, 'eval_accuracy': 0.6322701688555347, 'eval_runtime': 2.6918, 'eval_samples_per_second': 396.011, 'eval_steps_per_second': 6.315}

python3 main.py lora
{'eval_loss': 0.3925408720970154, 'eval_accuracy': 0.8367729831144465, 'eval_runtime': 3.805, 'eval_samples_per_second': 280.156, 'eval_steps_per_second': 4.468}
```