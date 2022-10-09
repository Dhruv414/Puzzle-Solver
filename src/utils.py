import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import edge_heuristic

permutations = [
    "0123",
    "0132",
    "0213",
    "0231",
    "0312",
    "0321",
    "1023",
    "1032",
    "1203",
    "1230",
    "1302",
    "1320",
    "2013",
    "2031",
    "2103",
    "2130",
    "2301",
    "2310",
    "3012",
    "3021",
    "3102",
    "3120",
    "3201",
    "3210"
]


def _img_to_array(img_path):
    # Load the image
    img = load_img(f'{img_path}', target_size=(128, 128))
    # Converts the image to a 3D numpy array (128x128x3)
    return np.asarray(img_to_array(img))


def get_pieces(img, rows, cols, row_cut_size, col_cut_size):
    pieces = []
    for r in range(0, rows, row_cut_size):
        for c in range(0, cols, col_cut_size):
            pieces.append(img[r:r + row_cut_size, c:c + col_cut_size, :])
    return pieces


# Splits an image into uniformly sized puzzle pieces
def get_uniform_rectangular_split(img, puzzle_dim_x, puzzle_dim_y):
    rows = img.shape[0]
    cols = img.shape[1]
    if rows % puzzle_dim_y != 0 or cols % puzzle_dim_x != 0:
        print('Please ensure image dimensions are divisible by desired puzzle dimensions.')
    row_cut_size = rows // puzzle_dim_y
    col_cut_size = cols // puzzle_dim_x

    pieces = get_pieces(img, rows, cols, row_cut_size, col_cut_size)

    return pieces


def permute_img(img, permutation):
    order = [int(x) for x in permutation]
    pieces = get_uniform_rectangular_split(img, 2, 2)
    return np.vstack((np.hstack((pieces[order[0]], pieces[order[1]])), np.hstack((pieces[order[2]], pieces[order[3]]))))


def permute_all(img):
    v = [0] * len(permutations)
    for i in range(len(permutations)):
        v[i] = permute_img(img, permutations[i])
    return v


def color_error(img, row1, col1, row2, col2):
    diff = [(img[row1][col1][i] - img[row2][col2][i]) ** 2 for i in range(3)]
    return sum(diff)


def get_filtered_permutations(img, K=1):
    mp = {}
    lst = set()
    # dictionary from int to list of strings
    for p in permutations:
        arr = permute_img(img, p)
        num_regions = get_number_regions(arr, TOLERANCE=850)
        lst.add(num_regions)
        if num_regions not in mp:
            mp[num_regions] = []
        mp[num_regions].append(p)
    lst = list(lst)
    lst.sort()
    ret = []
    for i in range(min(K, len(lst))):
        # which region # to use
        nreg = lst[i]
        for p in mp[nreg]:
            ret.append(p)
    return ret


def get_filtered_permutations_from_arr(pos_perms, img, K=1):
    mp = {}
    lst = set()
    # dictionary from int to list of strings
    for p in pos_perms:
        arr = permute_img(img, p)
        num_regions = get_number_regions(arr, TOLERANCE=850)
        lst.add(num_regions)
        if num_regions not in mp:
            mp[num_regions] = []
        mp[num_regions].append(p)
    lst = list(lst)
    lst.sort()
    ret = []
    for i in range(min(K, len(lst))):
        # which region # to use
        nreg = lst[i]
        for p in mp[nreg]:
            ret.append(p)
    return ret


def get_filtered_cross_heuristic_from_arr(pos_perms, img, k=1):
    mp = {}
    lst = set()
    # dictionary from int to list of strings
    for p in pos_perms:
        arr = permute_img(img, p)
        e_heuristic = edge_heuristic.cross_heuristic(arr)
        lst.add(e_heuristic)
        if e_heuristic not in mp:
            mp[e_heuristic] = []
        mp[e_heuristic].append(p)
    lst = list(lst)
    lst.sort()
    ret = []
    for i in range(min(k, len(lst))):
        # which region # to use
        nreg = lst[i]
        for p in mp[nreg]:
            ret.append(p)
    return ret


# Given an image, outputs the number of regions based on some tolerance for color rgb
# uses bfs
def get_number_regions(img, TOLERANCE):
    from collections import deque

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(x, y):
        return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

    rows = img.shape[0]
    cols = img.shape[1]
    g = [[0 for x in range(rows)] for y in range(cols)]
    visited = [[False for x in range(rows)] for y in range(cols)]

    regions = 0
    for row in range(rows):
        for col in range(cols):
            if not visited[row][col]:
                visited[row][col] = True
                q = deque()
                q.append((row, col))
                regions += 1

                while q:
                    r, c = q.popleft()
                    g[r][c] = regions
                    for dr, dc in dirs:
                        nr = r + dr
                        nc = c + dc
                        if in_bounds(nr, nc) and color_error(img, r, c, nr, nc) <= TOLERANCE and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))

    return regions
