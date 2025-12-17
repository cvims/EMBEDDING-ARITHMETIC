
# Embedding Arithmetic: Mitigating Bias in T2I Models

This repository contains the official code for the WACV26 paper "Embedding Arithmetic: Mitigating Bias in T2I Models with a Lightweight, Tuning-Free Framework". The framework enables bias mitigation in text-to-image (T2I) diffusion models using simple vector arithmetic in the embedding space, without the need for model fine-tuning.

## Features
- Lightweight, tuning-free bias mitigation for T2I models
- Works with popular diffusion pipelines (e.g., FLUX, Stable Diffusion)
- Easily extensible to new attributes and professions

## Setup.
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. (Recommended) The code needs to be run on a GPU machine with atleast 40GB VRAM to hold the FLUX model.

## Usage
Run the main notebook on the new environment to do some sample inferences and generate images with diverse demography:


The notebook demonstrates how to:
- Compute semantic attribute vectors for professions, race, and gender
- Apply embedding arithmetic to modify prompt embeddings
- Generate and visualize debiased images using the FLUX pipeline

## Files
- `embedding_arithmetic.ipynb`: Main notebook for experiments and visualization
- `flux_custom.py`: Custom pipeline for FLUX diffusion model
- `VLMAdapter.py`: Adapter modules for text encoders
- `requirements.txt`: Python dependencies


# Full Code Availability

Full code will be released upon acceptance.
