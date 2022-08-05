import sys
import json

from src.upload import upload
from src.model import build_model
from src.data import data
from src.train import train_model


def main(targets):

    with open('config/data_cfg.json') as f:
        data_cfg = json.load(f)

    with open('config/train_cfg.json') as f:
        train_cfg = json.load(f)


    if 'upload' in targets:
        upload()

    if 'data' in targets:
        train, val, test = data(**data_cfg)

    if 'model' in targets:
        model = build_model()
    
    if 'train' in targets:
        train_model(model, train, val, test, **train_cfg)



if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
