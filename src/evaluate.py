import wandb
import keras
import numpy as np


def evaluate(project_name, model_name, test_data):
    run = wandb.init(project=project_name, job_type='inference')

    model_at = run.use_artifact(model_name)
    model_dir = model_at.download()
    model = keras.models.load_model(model_dir)

    test = np.load(test_data)
    loss, accuracy = model.evaluate(test['x'], test['y'])

    return loss, accuracy
