import tensorflow as tf
import wandb
import numpy as np
from collections import namedtuple


def upload():

    datasets = make_datasets()
    names = ["training", "validation", "test"]

    run = wandb.init(project="leopard", job_type="upload-data")

    artifact = wandb.Artifact(
        name="mnist",
        type="dataset",
        description="MNIST dataset split into train/val/test",
        metadata={
            "source": "keras.datasets.mnist",
            "sizes": [len(dataset.x) for dataset in datasets]
        }
    )
    
    for name, data in zip(names, datasets):
        with artifact.new_file(name + ".npz", mode="wb") as file:
            np.savez(file, x=data.x, y=data.y)

    # save to W&B
    run.log_artifact(artifact)




def make_datasets(train_size=50_000):
    Dataset = namedtuple('Dataset', ['x', 'y'])

    # load
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # preprocess
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # train/val split
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    # split
    train = Dataset(x_train, y_train)
    val = Dataset(x_val, y_val)
    test = Dataset(x_test, y_test)

    datasets = (train, val, test)

    return datasets
