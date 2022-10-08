import numpy as np


def sum_rgb(rgb):
    return rgb[0] + rgb[1] + rgb[2]


def edge_heuristic(img_arr):
    # we want to split into 4 sub images corresponding to each square
    # calculate the difference of the edges of all connected edges
    # if two images, a and b, calculate according to formula: (RGB(a) - RGB(b))^2
    # Look at seam carving algorithm for a good heuristic
    if len(img_arr) == 0:
        print("Something terrible happened")
        return 0
    t = [0] * 4

    # 7.88408603e+160 8.92387900e+006 7.96534500e+006 6.24374300e+006
    img_height = len(img_arr) // 2
    img_width = len(img_arr[0]) // 2

    for i in range(int(img_height)):
        t[0] = t[0] + pow(sum_rgb(img_arr[i][img_width - 1]) - sum_rgb(img_arr[i][img_width]), 2)
        t[1] = t[1] + pow(sum_rgb(img_arr[img_width - 1 + i][img_width - 1])
                          - sum_rgb(img_arr[img_width - 1 + i][img_width]), 2)

        t[2] = t[2] + pow(sum_rgb(img_arr[img_height - 1][i]) - sum_rgb(img_arr[img_width][i]), 2)
        t[3] = t[3] + pow(sum_rgb(img_arr[img_height - 1][img_width - 1 + i])
                          - sum_rgb(img_arr[img_height][img_width - 1 + i]), 2)

    return np.sum(t)
