import os

import torch
import yaml

from model import SEDTask4_2021
from nnet.CRNN import CRNN
from utils.encoder import ManyHotEncoder

MODEL_BASE_DIR = "./result"
# MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=197-step=20591.ckpt")  # Speech and 9 other classes (10 classes)
MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=177-step=6407.ckpt")  # Speech only

# Load config from hparams.yaml
CONFIG_FILE_PATH = os.path.join(MODEL_BASE_DIR, "hparams.yaml")
with open(CONFIG_FILE_PATH, "r") as f:
    config = yaml.safe_load(f)

labels = [
    "Alarm_bell_ringing",
    "Blender",
    "Cat",
    "Dishes",
    "Dog",
    "Electric_shaver_toothbrush",
    "Frying",
    "Running_water",
    "Speech",
    "Vacuum_cleaner",
]


def load_model():
    MODEL_BASE_DIR = "./result"
    # MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=197-step=20591.ckpt")  # Speech and 9 other classes (10 classes)
    MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=177-step=6407.ckpt")  # Speech only

    # Load config from hparams.yaml
    CONFIG_FILE_PATH = os.path.join(MODEL_BASE_DIR, "hparams.yaml")
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    labels = [
        "Alarm_bell_ringing",
        "Blender",
        "Cat",
        "Dishes",
        "Dog",
        "Electric_shaver_toothbrush",
        "Frying",
        "Running_water",
        "Speech",
        "Vacuum_cleaner",
    ]


    map_location='cpu'
    #if torch.cuda.is_available():
    #    map_location=lambda storage, loc: storage.cuda()
    #else:
     #   map_location='cpu'
    #print(map_location)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    test_model_state_dict = checkpoint["state_dict"]
    sed_student = torch.nn.DataParallel(
        CRNN(**config["net"]).to(device="cuda" if torch.cuda.is_available() else "cpu")
        # CRNN(**config["net"]).to(device="cpu")

    )
    encoder = ManyHotEncoder(
        labels=labels,
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    desed_training = SEDTask4_2021(
        hparams=config,
        encoder=encoder,
        sed_student=sed_student,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # device=torch.device("cpu"),

    )
    desed_training.load_state_dict(test_model_state_dict)
    print("Loaded!")

    return desed_training
