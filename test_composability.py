"""
Hypothesis 2: Composability.

Generates images for an "intersectional" prompt embedding (profession + the
directly-measured direction for e.g. "asian female") side by side with a
"composed" prompt embedding (profession + the sum of the individual "asian"
and "female" direction vectors). If composability holds, the two sets of
images should be visually and semantically similar, since the composed
direction is expected to approximate the intersectional one.
"""
import re
import os
import argparse
import torch
from tqdm import tqdm

from VLMAdapter import PrismVLMTextEncoder
from flux_custom import FluxCustomPipeline

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

PROFESSIONS = ["Nurse", "Doctor", "Engineer", "Teacher", "Scientist", "Chef", "Police Officer", "CEO", "Artist", "Construction Worker"]
RACE_ATTRS = ["white", "black", "asian", "indian"]
GENDER_ATTRS = ["male", "female"]


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


def main(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    pipe = FluxCustomPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    ).to("cuda")
    prism = PrismVLMTextEncoder("openai/clip-vit-large-patch14", adapter_type="identity")

    intersections = [f"{race} {gender}" for race in RACE_ATTRS for gender in GENDER_ATTRS]

    semantic_vectors = create_semantic_vector_lookup(PROFESSIONS, RACE_ATTRS, GENDER_ATTRS, prism)
    semantic_averages = average_semantic_attribute_vectors(semantic_vectors)

    composed_dir = os.path.join(args.output_dir, "Composed")
    intersectional_dir = os.path.join(args.output_dir, "Intersectional")

    for p in PROFESSIONS:
        prompt = f"a photo portrait of {choose_article(p)} {p}"
        print(f"Generating for prompt : {prompt}")
        prompt_embed = prism(prompt)
        for inter in intersections:
            composed_imdir = os.path.join(composed_dir, p, inter)
            intersectional_imdir = os.path.join(intersectional_dir, p, inter)

            os.makedirs(composed_imdir, exist_ok=True)
            os.makedirs(intersectional_imdir, exist_ok=True)

            race, gender = inter.split(" ")
            inter_direction = semantic_averages[inter]
            composed_direction = semantic_averages[race] + semantic_averages[gender]

            inter_prompt = prompt_embed + inter_direction
            composed_prompt = prompt_embed + composed_direction

            for i, s in enumerate(SEEDS):
                generator = torch.Generator(device="cuda").manual_seed(s)
                composed_image_path = os.path.join(composed_imdir, f"Image_{i}_{s}.png")
                intersectional_image_path = os.path.join(intersectional_imdir, f"Image_{i}_{s}.png")

                image_inter, _ = pipe.generate_with_custom_embedding(inter_prompt, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator)
                image_composed, _ = pipe.generate_with_custom_embedding(composed_prompt, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator)

                image_inter.save(intersectional_image_path)
                image_composed.save(composed_image_path)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composability hypothesis: composed vs. intersectional direction vectors")
    parser.add_argument("--output_dir", type=str, default="./results/composability", help="Root directory to write generated images to")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), help="HuggingFace cache directory")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace access token (or set the HF_TOKEN env var). Required for gated FLUX.1-dev weights.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (only used if CUDA_VISIBLE_DEVICES isn't already set)")
    parser.add_argument("--steps", type=int, default=26, help="Number of diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parsed_args = parser.parse_args()
    if not parsed_args.hf_token:
        raise SystemExit("A HuggingFace token is required: pass --hf_token or set the HF_TOKEN environment variable.")
    main(parsed_args)
