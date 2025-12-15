# Deep Learning Coursework

## Healthcare AI Assistant

This repository contains the implementation of a **Healthcare AI Assistant** developed for the Deep Learning coursework.  
The project integrates **computer vision** and **language models** to analyse visual data, explain predictions, and evaluate performance, robustness, and sustainability.

---

## Project Overview

This coursework explores:
- Convolutional Neural Networks trained **from scratch**
- The effect of **data augmentation**
- **L1 vs L2 regularisation**
- **Transfer learning** using ResNet-18
- Dataset complexity analysis
- Model **robustness**, **interpretability (Grad-CAM)**, and **sustainability**
- A core **LLM** evaluated using perplexity and BLEU

All experiments are implemented in **Jupyter notebooks** and are fully reproducible.

---

## Repository Structure

- **visionmodel/**
  - **customcnn/**
    - **augmentationvsnoaugmentation/**
      - **augmentation/**
        - `augmentationcode.ipynb`
      - **noaugmentation/**
        - `noaugmentationcode.ipynb`
    - **l1vsl2/**
      - `l1_l2_code.ipynb`
  - **resnet-18/**
    - **fruit360-dataset/**
      - `pretrained.ipynb`
    - **openimagesdataset/**
      - `pretrained_open.ipynb`

- **languagemodel/**
  - **corellmliteracy/**
    - `llm.ipynb`

- **inference/**
  - **objectdetection/**
    - `app.py`



## Vision Models

### Custom CNN
- Augmentation vs No Augmentation
- L1 vs L2 regularisation
- Training and validation curves
- Confusion matrices
- Robustness testing (brightness and occlusion)
- Grad-CAM explainability

### ResNet-18 (Transfer Learning)
- Fine-tuned on Fruit-360
- Fine-tuned on OpenImages
- Demonstrates the effect of dataset complexity on generalisation

---

## Language Model

- Core LLM trained for AI literacy
- Evaluated using:
  - Validation loss
  - Perplexity
  - BLEU-2 score
- Includes runtime and energy consumption analysis

---

## Running the Code

### Clone the repository
```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant

#Install dependencies
pip install -r requirements.txt
```
All notebooks are executed using Jupyter, with each .ipynb run from top to bottom to automatically generate training curves, performance metrics (Accuracy, Macro-F1, BLEU, Perplexity), confusion matrices, and Grad-CAM visualisations.
The project also evaluates sustainability by analysing FLOPs, parameter counts, runtime, and energy consumption, highlighting performance–compute trade-offs for responsible AI development.

Author: Group 10
