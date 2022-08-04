import sys
import json

from src.upload import upload
from src.model import build_model
from src.data import data
from src.train import train_model
from src.evaluate import evaluate


def main(targets):
    with open('config/train_cfg.json') as f:
        train_cfg = json.load(f)

    with open('config/data_cfg.json') as f:
        data_cfg = json.load(f)

    with open('config/evaluate_cfg.json') as f:
        evaluate_cfg = json.load(f)


    if 'upload' in targets:
        upload()
    
    if 'model' in targets:
        model = build_model()

    if 'data' in targets:
        train, val, test = data(**data_cfg)
    
    if 'train' in targets:
        train_model(model, train, val, **train_cfg)
    
    if 'evaluate' in targets:
        evaluate(**evaluate_cfg)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
