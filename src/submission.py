# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
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

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.model = load_model('example_model.h5')

    def make_prediction(self, img_path):
        arr = utils._img_to_array(img_path)

        ve = utils.permute_all(arr)
        pos = utils.get_filtered_permutations(arr, 3)
        print(pos)
        v = [0] * len(pos)
        for h in range(len(pos)):
            fl = utils.permute_img(arr, pos[h])
            v[h] = edge_heuristic.edge4_heuristic(fl)

        # print(utils.get_number_regions(arr, TOLERANCE=20000))

        perm_arr = utils.permute_img(arr, utils.permutations[np.argmin(v)])
        perm_img = Image.fromarray((perm_arr * 255).astype(np.uint8))

        # perm_img.show()
        return pos[np.argmin(v)]


# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':
    i = 0
    g = 0
    ANS = "3210"
    LIMIT = 30
    SHOULD_STOP = True

    for img_name in glob('3210/*'):
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
