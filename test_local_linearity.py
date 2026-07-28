"""
Hypothesis 3: Local Linearity.

Scales a single semantic attribute direction vector (e.g. "asian female") by a
sweep of alpha coefficients added on top of a fixed profession prompt
embedding, generates an image for each (seed, alpha) pair, and scores each
image with (a) a FairFace/MTCNN gender-confidence detector and (b) a BLIP-VQA
semantic-consistency score. If the relationship between alpha and the
attribute's measured presence is locally linear, gender confidence should
increase roughly monotonically with alpha over the tested range, without a
large loss in semantic (profession) consistency.
"""
import re
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering

from VLMAdapter import PrismVLMTextEncoder
from flux_custom import FluxCustomPipeline
from get_face_attributes_facenet import load_models, load_transforms, detect_race_gender

SEEDS = [
    598154815, 1066353334, 1998757406, 1696943126, 1861108563,
    721691552, 1399725696, 692997605, 408797720, 321738267,
    1551422566, 1201384180, 473842648, 1430062913, 1050387955,
    1608382366, 1495710670, 1154098134, 1536228832, 1964958588,
    1656241920, 1647649170, 743743653, 317709255, 798993664,
    1111375164, 331196234, 1303817740, 2046687807, 132013211,
    1238418236, 1405797654, 2010767431, 271202386, 1450232097,
    330694197, 172531179, 1161840315, 2141291718, 2006992246,
    1141061462, 1520817402, 1935562663, 1045992321, 2122552747,
    1573312589, 2071881590, 1158266152, 764043458, 1835924590,
    1378913242, 1957393370, 135541314, 329282746, 190820242,
    1655845414, 198430532, 505184200, 131709356, 1838902176,
    1444240574, 2115460845, 1327682050, 2028357277, 801421117,
    2002336962, 621174870, 569259509, 1205078404, 642948010,
    1975892372, 215404229, 1039798279, 1729858859, 1342616620,
    393158281, 1676747224, 1820202943, 1013434885, 445305063,
    1693313335, 1854424984, 1115172966, 1738656838, 1536643082,
    414703234, 1548869964, 1898575321, 1791855621, 1668283477,
    2064688014, 513918664, 479428030, 552315579, 1644489163,
    332355964, 285249710, 1746194035, 1881229850, 145298052,
]

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.00, 3.25, 3.5, 4.0, 4.5, 5.0]

PROFESSIONS_FOR_VECTORS = ["Nurse", "Doctor", "Engineer", "Teacher", "Scientist", "Chef", "Police Officer", "CEO", "Artist", "Construction Worker"]
RACE_ATTRS = ["white", "black", "asian", "indian"]
GENDER_ATTRS = ["male", "female"]
GENDER_INDEX = {"male": 0, "female": 1}


def choose_article(phrase):
    first_word = re.findall(r"\b\w+\b", phrase)[0]
    return "An" if first_word[0].lower() in "aeiou" else "A"


def create_semantic_vector_lookup(professions, races, genders, encoder):
    prompts = {}
    for prof in professions:
        prompts[prof] = f"a photo portrait of {choose_article(prof)} {prof}"
        for race in races:
            prompts[f"{race} {prof}"] = f"a photo portrait of {choose_article(race)} {race} {prof}"
        for gender in genders:
            prompts[f"{gender} {prof}"] = f"a photo portrait of {choose_article(gender)} {gender} {prof}"
        for race in races:
            for gender in genders:
                prompts[f"{race} {gender} {prof}"] = f"a photo portrait of {choose_article(race)} {race} {gender} {prof}"

    print("\nCalculating embeddings for all prompts...")
    embeddings = {}
    for key, prompt in tqdm(prompts.items(), desc="Encoding Prompts"):
        embeddings[key] = encoder([prompt])

    semantic_vectors = {}
    intersection = [f"{race} {gender}" for race in races for gender in genders]
    all_attributes = races + genders + intersection

    for attr in tqdm(all_attributes):
        semantic_vectors[attr] = {}
        for prof in professions:
            base_embedding = embeddings[prof]
            attr_embedding = embeddings[f"{attr} {prof}"]
            semantic_vectors[attr][prof] = attr_embedding - base_embedding

    print("\n--- Semantic Vector Lookup Table Created Successfully ---")
    return semantic_vectors


def average_semantic_attribute_vectors(semantic_vectors):
    averages = {}
    for k in semantic_vectors.keys():
        embs = torch.cat([semantic_vectors[k][p] for p in semantic_vectors[k].keys()]).mean(dim=0)
        averages[k] = embs
    return averages


def gender_index_for_concept(concept):
    for token, idx in GENDER_INDEX.items():
        if token in concept.split():
            return idx
    raise ValueError(f"Could not infer a gender token (male/female) from concept '{concept}'")


def main(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = "cuda"

    pipe = FluxCustomPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    ).to(device)
    prism = PrismVLMTextEncoder("openai/clip-vit-large-patch14", adapter_type="identity", device=device)

    mtcnn, _, model_fair_7 = load_models(device=device, fairface_weights=args.fairface_weights)
    trans = load_transforms()

    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

    def get_vqa_score(image, profession):
        """Confidence (softmax prob of the 'yes' token) that the image matches the profession."""
        question = f"Is the person in this image {choose_article(profession)} {profession}?"
        inputs = vqa_processor(image, question, return_tensors="pt").to(device)
        outputs = vqa_model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
        yes_token_id = 2748  # Token ID for 'yes' in BLIP's vocabulary
        logits = outputs.scores[0][0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities[yes_token_id].item()

    def get_ensemble_consistency_score(image, profession):
        return get_vqa_score(image, profession)

    semantic_vectors = create_semantic_vector_lookup(PROFESSIONS_FOR_VECTORS, RACE_ATTRS, GENDER_ATTRS, prism)
    semantic_averages = average_semantic_attribute_vectors(semantic_vectors)

    concept_gender_index = gender_index_for_concept(args.concept)
    alphas = args.alphas or DEFAULT_ALPHAS

    gender_conf_scores = []
    consistency_scores = []

    prompt = f"a photo of {choose_article(args.profession)} {args.profession} on stage with an award"
    prompt_embedding = prism(prompt)

    for seed in tqdm(SEEDS):
        target_dir = os.path.join(args.output_dir, str(seed))
        os.makedirs(target_dir, exist_ok=True)
        seed_gender_conf_scores = []
        seed_consistency_scores = []
        for alp in alphas:
            generator = torch.Generator(device).manual_seed(seed)
            additive = prompt_embedding + alp * semantic_averages[args.concept]
            image, _ = pipe.generate_with_custom_embedding(additive, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator)
            image.save(os.path.join(target_dir, f"{str(alp).replace('.', '_')}.png"))
            try:
                result_dict = detect_race_gender(image, mtcnn, model_fair_7, trans, return_confidences=True, device=device)
            except Exception:
                result_dict = {"gender": None, "race": None, "gender_score": [None, None]}
            consistency_score = get_ensemble_consistency_score(image, args.profession)

            seed_gender_conf_scores.append(result_dict["gender_score"][concept_gender_index])
            seed_consistency_scores.append(consistency_score)

        consistency_scores.extend([{"alpha": a, "score": s, "metric": "consistency", "seed": seed} for a, s in zip(alphas, seed_consistency_scores)])
        gender_conf_scores.extend([{"alpha": a, "score": s, "metric": "gender_conf", "seed": seed} for a, s in zip(alphas, seed_gender_conf_scores)])

    df = pd.DataFrame(gender_conf_scores + consistency_scores)
    df.to_csv(os.path.join(args.output_dir, "consistency_scores.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Linearity hypothesis: sweep alpha over a single attribute direction")
    parser.add_argument("--profession", type=str, default="Film Director")
    parser.add_argument("--concept", type=str, default="asian female", help="Attribute key to scale, e.g. 'asian female', 'male', 'indian'")
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="Alpha values to sweep (defaults to a 0.0-5.0 sweep)")
    parser.add_argument("--output_dir", type=str, default="./results/local_linearity")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--fairface_weights", type=str, default=os.environ.get("FAIRFACE_WEIGHTS"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parsed_args = parser.parse_args()
    if not parsed_args.hf_token:
        raise SystemExit("A HuggingFace token is required: pass --hf_token or set the HF_TOKEN environment variable.")
    main(parsed_args)
