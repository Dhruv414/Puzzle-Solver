# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import sys

import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Import helper functions from utils.py
import utils
import edge_heuristic


class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def make_prediction(self, img_path):
        arr = utils._img_to_array(img_path)

        pos = utils.get_filtered_permutations(arr, 2)
        g = 5
        i = 0
        while len(pos) != 1 and i <= 8:
            print(pos)
            pas = []
            if i % 2 == 0:
                pas = utils.get_filtered_permutations_from_arr(pos, arr, K=g)
            else:
                pas = utils.get_filtered_cross_heuristic_from_arr(pos, arr, k=g)

            if not len(pas):
                # pick lower edge score
                # dont get raw number
                scores = np.array([edge_heuristic.cross_heuristic(utils.permute_img(arr, pos[x])) for x in range(len(pos))])
                pos = [pos[np.argmin(scores)]]

                print(pos)

                break
            i += 1
            g -= 1
            pos = pas

        # v = [0] * len(pos)
        # for h in range(len(pos)):
        #     fl = utils.permute_img(arr, pos[h])
        #     c = edge_heuristic.cross_heuristic(fl)
        #     print(c)
        #     if c <= LOWER_BOUND:
        #         v[h] = sys.maxsize
        #     else:
        #         v[h] = c
        #
        perm_arr = utils.permute_img(arr, pos[0])

        perm_img = Image.fromarray((perm_arr * 255).astype(np.uint8))

        perm_img.show()
        return pos[0]


# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':
    i = 0
    g = 0
    ANS = "1302"
    LIMIT = 30
    SHOULD_STOP = True

    for img_name in glob('1302/*'):
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
