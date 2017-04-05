import os
import sys
import numpy as np
from create_submission import create_submission

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

NB_AUGMENTATIONS = 5
MODEL_DIR = "models"
PREDICT_SCRIPT_NAME = "test.py"
OUTPUT_FILE = "temp_pred.npy"
NB_CLASSES = 8
NB_TEST_SAMPLES = 1000

if __name__ == "__main__":

    predictions = np.zeros((NB_TEST_SAMPLES, NB_CLASSES))
    models = os.listdir(MODEL_DIR)

    for model in models:
        model_path = os.path.join(MODEL_DIR, model)
        output_path = OUTPUT_FILE.replace(".npy", "_" + str(model) + ".npy")
        args = ["python", PREDICT_SCRIPT_NAME, model_path, output_path]
        command = " ".join(args)
        if os.system(command):
            raise RuntimeError("Failed to produce augmented predictions for current model.")

        result = np.load(OUTPUT_FILE)
        predictions += result

    predictions /= len(models)
    np.save("final.npy", predictions)

    print("Saving submission.")
    create_submission(predictions, "submit.csv")

