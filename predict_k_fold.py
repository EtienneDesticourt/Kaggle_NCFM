import os
import sys
import numpy as np
from create_submission import create_submission

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

WEIGHTS_DIR = "weights"
PREDICT_SCRIPT_NAME = "predict_average_augmentation.py"
OUTPUT_DIR = "temp2"

if __name__ == "__main__":
    predictions = None
    weights_files = os.listdir(WEIGHTS_DIR)
    for weights in weights_files:
        print("{} model predictions started.".format(weights))

        weights_path = os.path.join(WEIGHTS_DIR, weights)
        output_file = os.path.join(OUTPUT_DIR, weights + "_prediction.npy")

        # Gotta run it in a separate process because of tensorflow memory leak
        args = ["python", PREDICT_SCRIPT_NAME, weights_path, output_file]
        command = " ".join(args)
        if os.system(command):
            raise RuntimeError("Failed to produce augmented predictions for current model.")

        result = np.load(output_file)
        if predictions == None:
            predictions = result
        else:
            predictions += result


    predictions /= len(weights_files)
    np.save("final.npy", predictions)

    print("Saving submission.")
    create_submission(predictions, "submit.csv")
