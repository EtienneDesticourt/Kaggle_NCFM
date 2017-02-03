import numpy as np
import os
import sys

HEADER = 'image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n'
TEST_DIR = "test_stg1/New Folder/"

def create_submission(predictions, out_path):
    test_image_list = os.listdir(TEST_DIR)

    with open(out_path, 'w') as f:
        f.write(HEADER)
        for i, image_name in enumerate(test_image_list):
            pred = ['%.6f' % p for p in predictions[i, :]]
            row = ','.join(pred)
            f.write('%s,%s\n' % (image_name, row))

if __name__ == "__main__":

    predictions_path = sys.argv[1]
    out_path = sys.argv[2]

    predictions = np.load(predictions_path)
    create_submission(predictions, out_path)

