import numpy as np
import wandb
import os
from collections import namedtuple

def data(project, artifact):
    """
    :return: list of 3 Datasets [train, val, test]; access data with train.x train.y
    """

    run = wandb.init(project=project, job_type="data")
    data = run.use_artifact(artifact)
    data_dir = data.download()


    Dataset = namedtuple('Dataset', ['x', 'y'])
    datasets = []
    
    
    for dataset in ['training', 'validation', 'test']:

        data = np.load(os.path.join(data_dir, dataset + '.npz'))

        datasets.append(Dataset(x=data["x"], y=data["y"]))


    
    return datasets

