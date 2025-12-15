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
Run the notebooks
jupyter notebook


Open any .ipynb file and run all cells from top to bottom.

Output

When a notebook is executed:

Training and validation plots are generated automatically

Performance metrics (accuracy, Macro-F1, BLEU, perplexity) are computed

Confusion matrices and Grad-CAM visualisations are produced

Results are displayed and/or saved

Sustainability

The project evaluates:

FLOPs and parameter counts for vision models

Runtime and energy consumption for the LLM

Performance versus compute trade-offs

This supports responsible and sustainable AI development.
