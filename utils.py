import numpy as np

def get_pieces(img, rows, cols, row_cut_size, col_cut_size):
    pieces = []
    for r in range(0, rows, row_cut_size):
        for c in range(0, cols, col_cut_size):
            pieces.append(img[r:r+row_cut_size, c:c+col_cut_size, :])
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

def visualize_image(permutation: str):
  a, b, c, d = map(int, list(permutation))

  pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
  # Example images are all shuffled in the "3120" order
  final_image = Image.fromarray(np.vstack((np.hstack((pieces[a],pieces[b])),np.hstack((pieces[c],pieces[d])))))
  display(final_image)
