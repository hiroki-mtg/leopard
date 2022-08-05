from codecs import BOM_BE
from unittest.util import three_way_cmp
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def train_model(
    model,
    train, val, test,
    project,
    loss, optimizer, metric,
    epochs, batch_size
    ):

    # init
    run = wandb.init(project=project)

    # compile
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metric]
        )

    # fit
    model.fit(
        train.x,
        train.y,
        validation_data=(val.x, val.y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[WandbCallback()]
        )

    loss, accuracy = model.evaluate(test.x, test.y)
    model.save('my_model')



    import numpy as np
    artifact = wandb.Artifact(
        name='convnet',
        type = 'model',
        description='Trained convnet model',
        metadata={
            'epochs': epochs,
            'test loss': loss,
            'test accuracy': accuracy
            
        },

    )


    artifact.add_dir('my_model')
    run.log_artifact(artifact)
