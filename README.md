
# Embedding Arithmetic: Mitigating Bias in T2I Models

[![arXiv](https://img.shields.io/badge/arXiv-2604.18167-b31b1b.svg)](https://arxiv.org/pdf/2604.18167)

This repository contains the official code for the ICPR 2026 paper "Embedding Arithmetic: Mitigating Bias in T2I Models with a Lightweight, Tuning-Free Framework". The framework enables bias mitigation in text-to-image (T2I) diffusion models using simple vector arithmetic in the embedding space, without the need for model fine-tuning.

## Features
- Lightweight, tuning-free bias mitigation for T2I models
- Works with popular diffusion pipelines (e.g., FLUX, Stable Diffusion)
- Easily extensible to new attributes and professions

## Setup
1. Requires Python 3.12 (developed and tested on 3.12.11).
2. Install dependencies:
	```bash
	pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
	```
3. (Recommended) The code needs to be run on a GPU machine with atleast 40GB VRAM to hold the FLUX model.
4. Provide a HuggingFace access token with access to `black-forest-labs/FLUX.1-dev` (accept the model's gated license on HuggingFace first), either via the `HF_TOKEN` environment variable or the `--hf_token` flag accepted by every script below. Never commit a token to the repository.
5. For `test_local_linearity.py`, download the FairFace checkpoint `res34_fair_align_multi_7_20190809.pt` from [this folder](https://drive.google.com/drive/folders/1yUYaE5aRgNKCI5PuzeUAr0NWiS4_Rakc?usp=sharing) and point `--fairface_weights` (or the `FAIRFACE_WEIGHTS` env var) at it.

## Usage
`embedding_arithmetic.ipynb` demonstrates the core idea: computing semantic attribute vectors for professions, race, and gender, and adding them to prompt embeddings to generate images with diverse demography.

Beyond that demo, the paper evaluates four hypotheses about the embedding-arithmetic direction vectors. Each has a corresponding script/notebook below, all built on the same core idea: encode `"<profession>"` and `"<attribute> <profession>"` prompts with the frozen CLIP text encoder (`VLMAdapter.PrismVLMTextEncoder`), take the mean difference to get a per-attribute direction vector, then add a scaled direction vector to a prompt embedding before generating with FLUX (`flux_custom.FluxCustomPipeline.generate_with_custom_embedding`).

### Hypothesis 1 — Orthogonality (`hypothesis_orthogonality.ipynb`)
Are the direction vectors for different demographic attributes (gender, race, and their intersections) approximately orthogonal to one another? The notebook builds the direction vectors and plots pairwise cosine-similarity heatmaps; near-zero off-diagonal values support the hypothesis.
```bash
jupyter nbconvert --to notebook --execute hypothesis_orthogonality.ipynb
```

### Hypothesis 2 — Composability (`test_composability.py`)
Does adding two individual attribute vectors (e.g. "asian" + "female") approximate the directly-measured intersectional vector (e.g. "asian female")? Generates paired "composed" vs. "intersectional" image sets for each profession.
```bash
python test_composability.py --output_dir ./results/composability --hf_token $HF_TOKEN
```

### Hypothesis 3 — Local Linearity (`test_local_linearity.py`)
Does scaling a single attribute direction vector by increasing alpha coefficients produce a locally linear (roughly monotonic) increase in the measured attribute, without breaking semantic consistency? Sweeps alpha for one profession/concept pair and scores every generated image for gender confidence (FairFace) and profession consistency (BLIP-VQA).
```bash
python test_local_linearity.py --profession "Film Director" --concept "asian female" \
    --output_dir ./results/local_linearity --hf_token $HF_TOKEN --fairface_weights /path/to/res34_fair_align_multi_7_20190809.pt
```

### Hypothesis 4 — Generalisability (`test_embedding_arithmetic.py`)
Do direction vectors learned from one set of professions still steer demographic presentation correctly on entirely different, held-out professions/scenes? Applies a fixed-alpha direction vector to held-out professions (Firefighter, Judge, Pilot, ...).
```bash
python test_embedding_arithmetic.py --queue gender --output_dir ./results/generalisability --hf_token $HF_TOKEN
```

### Metric computation
- `get_consistency_score.py` computes the Concept Consistency Score (CCS): an ensemble of VQA models (BLIP, GIT-base, GIT-large, ViLT) each answer "Is the person in this image a/an `<profession>`?" for every image in a `<img_dir>/<profession>/*` tree, and writes `concept_score.csv`.
  ```bash
  python get_consistency_score.py --img_dir ./results/composability/Composed
  ```
- `get_face_attributes_facenet.py` runs MTCNN face detection + FairFace race/gender classification over an `ImageFolder`-style directory and writes `attributes_fairface_with_facenet.csv`. This is also used internally by `test_local_linearity.py`.
  ```bash
  python get_face_attributes_facenet.py --image_dir ./results/generalisability/gender --fairface_weights /path/to/res34_fair_align_multi_7_20190809.pt
  ```

## Files
- `embedding_arithmetic.ipynb`: Main notebook for experiments and visualization
- `hypothesis_orthogonality.ipynb`: Hypothesis 1 (Orthogonality)
- `test_composability.py`: Hypothesis 2 (Composability)
- `test_local_linearity.py`: Hypothesis 3 (Local Linearity)
- `test_embedding_arithmetic.py`: Hypothesis 4 (Generalisability)
- `get_consistency_score.py`: Concept Consistency Score (CCS) metric computation
- `get_face_attributes_facenet.py`: Face/race/gender attribute detection metric computation
- `flux_custom.py`: Custom pipeline for FLUX diffusion model
- `VLMAdapter.py`: Adapter modules for text encoders
- `requirements.txt`: Python dependencies
