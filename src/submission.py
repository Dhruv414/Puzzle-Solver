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

        v = [0] * 24
        ve = utils.permute_all(arr)
        for i in range(len(ve)):
            v[i] = edge_heuristic.edge_heuristic(ve[i])

        perm_arr = utils.permute_img(arr, utils.permutations[np.argmin(v)])
        perm_img = Image.fromarray((perm_arr * 255).astype(np.uint8))

        perm_img.show()
        return utils.permutations[np.argmin(v)]


# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    i = 0
    for img_name in glob('images/*'):
        # Open an example image using the PIL library
        if i == 10:
            break
        predictor = Predictor()
        pred = predictor.make_prediction(img_name)

        print(pred)

        i = i + 1
