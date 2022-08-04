import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def train_model(
    model,
    train, val,
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

    artifact = wandb.Artifact(
        name='convnet',
        type = 'model',
        description='Trained convnet model'
    )

    model.save('my_model')
    artifact.add_dir('my_model')
    run.log_artifact(artifact)
