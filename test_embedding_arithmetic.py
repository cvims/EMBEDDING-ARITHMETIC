"""
Hypothesis 4: Generalisability.

Applies a fixed-magnitude attribute direction vector (gender, race, or a
race+gender intersection) on top of prompt embeddings for professions that
were *not* used to compute the direction vectors themselves (e.g.
"Firefighter", "Judge", "Pilot", ...). Each of the held-out professions is
paired with a pseudo-random target attribute queue so that generation isn't
biased toward a single demographic. If embedding arithmetic generalises, the
same additive direction vectors learned from one set of professions should
still steer demographic presentation correctly on entirely different
professions/scene descriptions.
"""
import re
import os
import argparse
import torch
from tqdm import tqdm

from VLMAdapter import PrismVLMTextEncoder
from flux_custom import FluxCustomPipeline

PROFESSIONS_FOR_VECTORS = ["Nurse", "Doctor", "Engineer", "Teacher", "Scientist", "Chef", "Police Officer", "CEO", "Artist", "Construction Worker"]
RACE_ATTRS = ["white", "black", "asian", "indian"]
GENDER_ATTRS = ["male", "female"]

HELD_OUT_PROFESSIONS = ["Firefighter", "Judge", "Pilot", "Astronaut", "Farmer", "Flight Attendant", "Mechanic", "Carpenter"]
HELD_OUT_ACTIONS = [
    "standing in front of a fire truck, holding a helmet", "sitting at the bench in a courtroom, looking forward",
    "standing in front of an airplane, smiling at the camera", "posing in a space suit",
    "standing in a field with crops, holding a basket of produce", "standing in the airplane aisle, smiling at passengers",
    "standing in a garage, holding a wrench and looking at the camera", "standing at a workbench, holding a piece of wood",
]

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

GENDER_QUEUE = ['female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male',
                'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male',
                'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female',
                'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female']

RACE_QUEUE = ['asian', 'white', 'black', 'indian', 'asian', 'white', 'white', 'black', 'asian', 'indian',
              'white', 'asian', 'black', 'white', 'indian', 'asian', 'white', 'black', 'white', 'indian',
              'black', 'asian', 'white', 'black', 'indian', 'white', 'asian', 'black', 'white', 'asian',
              'indian', 'white', 'black', 'asian', 'indian', 'white', 'black', 'asian', 'white', 'indian',
              'black', 'asian', 'white', 'black', 'indian', 'asian', 'white', 'black', 'asian', 'white',
              'asian', 'white', 'black', 'indian', 'asian', 'white', 'white', 'black', 'asian', 'indian',
              'white', 'asian', 'black', 'white', 'indian', 'asian', 'white', 'black', 'white', 'indian',
              'black', 'asian', 'white', 'black', 'indian', 'white', 'asian', 'black', 'white', 'asian',
              'indian', 'white', 'black', 'asian', 'indian', 'white', 'black', 'asian', 'white', 'indian',
              'black', 'asian', 'white', 'black', 'indian', 'asian', 'white', 'black', 'asian', 'white']

INTERSECTIONAL_QUEUE = [
    'black female', 'asian male', 'white male', 'indian female', 'white female', 'asian female', 'black male', 'white male',
    'white male', 'black female', 'asian male', 'indian male', 'white female', 'white female', 'black male', 'indian female',
    'asian female', 'white male', 'black female', 'white female', 'indian male', 'asian male', 'white male', 'black male',
    'asian female', 'white female', 'black female', 'indian male', 'white male', 'asian female', 'white female', 'black male',
    'indian female', 'asian male', 'white male', 'black female', 'white female', 'asian female', 'indian male', 'white male',
    'black male', 'asian female', 'white female', 'indian female', 'white male', 'black female', 'asian male', 'white female',
    'indian male', 'black male', 'white male', 'asian female', 'white female', 'black female', 'indian female', 'white male',
    'asian male', 'white female', 'black male', 'indian female', 'white male', 'asian female', 'black female', 'white female',
    'indian male', 'white male', 'asian male', 'black female', 'white female', 'indian female', 'white male', 'black male',
    'asian female', 'white female', 'indian male', 'white male', 'black female', 'asian male', 'white female', 'indian female',
    'white male', 'black male', 'asian female', 'white female', 'indian male', 'white male', 'black female', 'asian male',
    'white female', 'indian female', 'white male', 'black male', 'asian female', 'white female', 'indian male', 'white male',
    'black female', 'asian male', 'white female', 'indian female', 'white male', 'black male', 'asian female', 'white female',
]

QUEUES = {"gender": GENDER_QUEUE, "race": RACE_QUEUE, "intersectional": INTERSECTIONAL_QUEUE}


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

    semantic_vectors = create_semantic_vector_lookup(PROFESSIONS_FOR_VECTORS, RACE_ATTRS, GENDER_ATTRS, prism)
    semantic_averages = average_semantic_attribute_vectors(semantic_vectors)

    queue = QUEUES[args.queue]

    for p, a in zip(HELD_OUT_PROFESSIONS, HELD_OUT_ACTIONS):
        target_dir = os.path.join(args.output_dir, args.queue, p)
        prompt = f"a photo of {choose_article(p)} {p}, {a}"
        os.makedirs(target_dir, exist_ok=True)
        print(f"Generating for prompt : {prompt}")
        for i, (s, q) in tqdm(enumerate(zip(SEEDS, queue))):
            image_file = os.path.join(target_dir, f"Image_{i}_{s}.png")
            generator = torch.Generator(device="cuda").manual_seed(s)

            embedding = prism(prompt)
            semantic_direction = semantic_averages[q]
            additive = embedding + args.alpha * semantic_direction

            image, _ = pipe.generate_with_custom_embedding(additive, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator)
            image.save(image_file)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalisability hypothesis: apply direction vectors to held-out professions")
    parser.add_argument("--queue", type=str, default="gender", choices=list(QUEUES.keys()), help="Which target-attribute queue to pair with the seeds")
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--output_dir", type=str, default="./results/generalisability")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parsed_args = parser.parse_args()
    if not parsed_args.hf_token:
        raise SystemExit("A HuggingFace token is required: pass --hf_token or set the HF_TOKEN environment variable.")
    main(parsed_args)
