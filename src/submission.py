# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
from time import perf_counter

import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Import helper functions from utils.py
import utils
import edge_heuristic

"""
DO NOT RENAME THIS CLASS
This class enables automated judging
This class should stay named as `Predictor`
"""
from time import perf_counter
class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def make_prediction(self, img_path):
        BOUND = 500
        arr = utils._img_to_array(img_path)

        t1 = perf_counter()
        ve = [(edge_heuristic.edge_heuristic(perm_arr), perm) for perm_arr, perm in
              zip(utils.permute_all(arr), utils.permutations)]
        ve = [(score, perm) for score, perm in ve if score > BOUND]
        ve = sorted(ve)[:9]  # Get the top 9 scores

        t2 = perf_counter()
        pos = utils.get_filtered_permutations(arr, K=2, TOL=100, PERMUTATIONS=[b for a, b in ve])

        v = [0] * len(pos)
        for h in range(len(pos)):
            fl = utils.permute_img(arr, pos[h])
            c = edge_heuristic.edge_heuristic(fl)

        print(min(v), pos[np.argmin(v)])
        return pos[np.argmin(v)]

# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':
    i = 0
    g = 0
    ANS = "1302"
    LIMIT = 5
    SHOULD_STOP = False

    for img_name in glob('example_images/*'):
        if i == LIMIT and SHOULD_STOP:
            break
        predictor = Predictor()
        pred = predictor.make_prediction(img_name)
        if pred == ANS:
            g = g + 1
        print("img:", img_name, "pred:", pred)
        i = i + 1

    print("SUCCESS:", g)
    print("OUT OF:", i)
    print("RATE:", g / i)