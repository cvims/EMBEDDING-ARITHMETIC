"""
Metric script: Concept Consistency Score (CCS).

Walks a directory of generated images (organised as <img_dir>/<profession>/*.png,
the layout produced by test_composability.py, test_embedding_arithmetic.py, and
test_local_linearity.py) and scores each image with an ensemble of VQA models
answering "Is the person in this image <a/an> <profession>?". Writes
concept_score.csv with one row per image and one column per VQA model.
"""
import os
import re
import glob
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering, AutoModelForCausalLM, AutoProcessor, ViltProcessor, ViltForQuestionAnswering
import torch.nn.functional as F
from PIL import Image


def choose_article(phrase):
    first_word = re.findall(r"\b\w+\b", phrase)[0]
    return "An" if first_word[0].lower() in "aeiou" else "A"


def get_git_vqa_score(image_path, profession, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    question = f"Answer only with a yes or no: Is the person in the image, looking like of doing the work {choose_article(profession)} {profession}?"

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    generated = model.generate(pixel_values=pixel_values.to(device), input_ids=input_ids.to(device), max_length=50, output_scores=True, return_dict_in_generate=True)
    logits = generated.scores[0][0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    return probabilities[yes_id].item()


def get_vilt_vqa_score(image_path, profession, processor, model):
    image = Image.open(image_path)
    question = f"Is the person in the image {choose_article(profession)} {profession}?"
    yes_id = 3

    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    scores = F.softmax(outputs.logits, dim=-1).squeeze()
    return scores[yes_id].item()


def get_blip_vqa_score(image_path, profession, processor, model, device):
    """Confidence (softmax prob of the 'yes' token) that the image matches the profession."""
    image = Image.open(image_path).convert("RGB")
    question = f"Answer only with a yes or no: Is the person in this image {choose_article(profession)} {profession}? "
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
    yes_token_id = 2748  # Token ID for 'yes' in BLIP's vocabulary
    logits = outputs.scores[0][0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities[yes_token_id].item()


def main(args):
    device = args.device

    blip_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

    git_vqa_processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
    git_vqa_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa").to(device)

    git_large_vqa_processor = AutoProcessor.from_pretrained("microsoft/git-large-textvqa")
    git_large_vqa_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textvqa").to(device)

    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    main_df = pd.DataFrame()
    professions = os.listdir(args.img_dir)
    for p in professions:
        p_path = os.path.join(args.img_dir, p)
        if not os.path.isdir(p_path):
            continue
        images = glob.glob(os.path.join(p_path, "*.*"))
        for i in tqdm(images, desc=p):
            df_row = {
                "file_name": os.path.split(i)[-1],
                "profession": p,
                "vilt_score": get_vilt_vqa_score(i, p, vilt_processor, vilt_model),
                "git_base_score": get_git_vqa_score(i, p, git_vqa_processor, git_vqa_model, device),
                "git_large_score": get_git_vqa_score(i, p, git_large_vqa_processor, git_large_vqa_model, device),
                "blip_score": get_blip_vqa_score(i, p, blip_vqa_processor, blip_vqa_model, device),
            }
            main_df = pd.concat([main_df, pd.DataFrame([df_row])])

    main_df.to_csv(os.path.join(args.img_dir, "concept_score.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Concept Consistency Score (CCS) over a directory of generated images")
    parser.add_argument("--img_dir", type=str, required=True, help="Root directory containing profession subfolders with images.")
    parser.add_argument("--device", type=str, default="cuda")
    parsed_args = parser.parse_args()
    main(parsed_args)
