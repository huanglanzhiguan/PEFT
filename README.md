# Intro
This is a project for the generative AI udacity nanodegree program. <br>
The goal of this project is to apply lightweight fine-tuning to a foundation model. <br>

- Load a pre-trained model and evaluate its performance.
  - Load a pretrained HF model
  - Load and preprocess a dataset
  - Evaluate the model on the dataset
- Perform PEFT using the pre-trained model.
  - Create a PEFT model
  - Train the PEFT model
  - Save the PEFT model
- Perform inference using the fine-tuned model and compare the results with the pre-trained model.
  - Load the PEFT model
  - Evaluate the PEFT model on the dataset

## How to evaluate a pre-trained model?
The foundation model usually has two parts:
- Body
- Head

Head is the top layer that can be changed depending on the task. For example: text generation, classification, <br>
translation, Q&A, etc. <br>
