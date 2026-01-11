# PyTorch_Computer_Vision

# Project Overview:
This repository contains a computer vision project built with PyTorch, focused on classifying images from the FashionMNIST dataset.
I implemented multiple architectures — from simple feedforward networks to a TinyVGG‑style CNN — and compared their performance on the FashionMNIST dataset (10 clothing categories, grayscale 28×28 images).

The project demonstrates my ability to design, train, and evaluate deep learning models, while showcasing end‑to‑end workflow skills: dataset handling, model building, training loops, evaluation metrics, and visualization.

## Features
Model Variants

model_without_ReLU: baseline feedforward network

model_with_ReLU: improved dense network

TinyVGG_CNN: convolutional neural network inspired by VGG

## Training & Evaluation

Custom training/testing loops with tqdm progress bars

## Visualization

Bar charts comparing model accuracies

Annotated plots showing exact accuracy values

## Reproducibility

Seeded experiments (torch.manual_seed(42))

Clear documentation of hyperparameters

Accuracy and loss tracking per epoch

Training time measurement for performance comparison

## Results
Model	Accuracy (%)	Training Time
model_without_ReLU	~83.43%	Fast
model_with_ReLU	~75.2%	Moderate
TinyVGG_CNN	~88.50%	Longer

## Tech Stack
Language: Python 3

Framework: PyTorch

Libraries: torchvision, matplotlib, pandas, tqdm

Dataset: FashionMNIST (10 clothing categories, grayscale 28×28 images)
