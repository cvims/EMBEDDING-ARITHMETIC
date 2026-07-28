"""
Face detection (MTCNN) + race/gender attribute classification (FairFace ResNet-34).

Used as a metric-computation dependency by test_local_linearity.py to score how
the predicted gender/race of generated faces shifts as an attribute direction
vector is scaled by alpha.

The FairFace checkpoint is not redistributed here. Download
`res34_fair_align_multi_7_20190809.pt` from
https://drive.google.com/drive/folders/1yUYaE5aRgNKCI5PuzeUAr0NWiS4_Rakc?usp=sharing
and pass its path via --fairface_weights (or the FAIRFACE_WEIGHTS env var).
"""
import os
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
import torch.nn as nn
from tqdm import tqdm

# We merge the 2 asian classes (South-East & East) together, and ignore the
# Latino/Hispanic and Middle Eastern classes, considering low performance in these cases.
race_dict = {
    0: "white", 1: "black", 2: None,
    3: "asian", 4: "asian",
    5: "indian", 6: None,
}
race_dict_original = {
    0: "white", 1: "black", 2: "latino/hispanic",
    3: "east asian", 4: "southeast asian",
    5: "indian", 6: "middle eastern",
}

gender_dict = {
    0: "male", 1: "female"
}


def collate_fn(x):
    return x[0]


def load_models(device="cuda", fairface_weights=None):
    fairface_weights = fairface_weights or os.environ.get("FAIRFACE_WEIGHTS")
    if not fairface_weights:
        raise ValueError(
            "FairFace weights path not provided. Pass --fairface_weights or set the "
            "FAIRFACE_WEIGHTS env var. See module docstring for the download link."
        )

    mtcnn = MTCNN(
        image_size=224,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.5,
        post_process=False,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    model_fair_7 = models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load(fairface_weights, map_location=device))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()
    return mtcnn, resnet, model_fair_7


def load_transforms():
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return trans


def detect_race_gender(image, mtcnn, model_fair_7, trans, return_confidences=False, device="cuda"):
    """
    Given a PIL image: detect the face with MTCNN, crop it, transform it for the
    FairFace model, and predict gender/race (optionally with confidences).
    """
    x_aligned, prob, lm = mtcnn.detect(image, landmarks=True)
    size = image.size
    box = x_aligned[0]
    box = [
        int(max(box[0], 0)),
        int(max(box[1], 0)),
        int(min(box[2], size[0])),
        int(min(box[3], size[1])),
    ]
    out = image.crop(box)
    preprocess = trans(np.array(out)).unsqueeze(dim=0).to(device)
    attrs = model_fair_7(preprocess)
    attr_out = attrs[0]
    race_outputs = attr_out[:7].detach().cpu().numpy()
    gender_outputs = attr_out[7:9].detach().cpu().numpy()
    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
    race_pred = np.argmax(race_score)
    gender_pred = np.argmax(gender_score)
    gender = gender_dict[gender_pred]
    race = race_dict[race_pred]
    if not return_confidences:
        return gender, race
    return {"gender": gender, "race": race, "gender_score": gender_score, "race_score": race_score}


def main(args):
    print("Initializing Models......")
    mtcnn, resnet, model_fair_7 = load_models(device=args.device, fairface_weights=args.fairface_weights)
    trans = load_transforms()
    main_df = pd.DataFrame()
    dataset = datasets.ImageFolder(args.image_dir)
    images = dataset.imgs
    profession_names = {i: c for c, i in dataset.class_to_idx.items()}
    for i, (x, y) in tqdm(enumerate(dataset)):
        prompt = profession_names[y]
        if args.remove_article:
            prompt = prompt.lower().replace("a ", "").replace("an ", "")
        try:
            gender, race = detect_race_gender(x, mtcnn, model_fair_7, trans, device=args.device)
        except Exception as e:
            print(e)
            gender, race = None, None
        df_row = pd.DataFrame([{
            "file_name": os.path.split(images[i][0])[-1],
            "prompt": prompt,
            "gender": gender,
            "race": race,
        }])
        main_df = pd.concat([main_df, df_row])
    main_df.to_csv(os.path.join(args.image_dir, "attributes_fairface_with_facenet.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run face attribute analysis on a directory of images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory containing profession subfolders with images.")
    parser.add_argument("--fairface_weights", type=str, default=os.environ.get("FAIRFACE_WEIGHTS"), help="Path to res34_fair_align_multi_7_20190809.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--remove_article", action="store_true", help="Strip leading article from the profession name / labels")
    parsed_args = parser.parse_args()
    main(parsed_args)
