import copy


def split_and_fill_image(array, normalized_array, resolution, small_units, transform, sigma):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    array = copy.deepcopy(array)
    normalized_array = copy.deepcopy(normalized_array)

    height, width = array.shape
    x_tiles = width // resolution + 1
    y_tiles = height // resolution + 1

    padding_x = (resolution * x_tiles - width) // 2
    padding_y = (resolution * y_tiles - height) // 2

    padding_x += (resolution * x_tiles - width) % 2
    padding_y += (resolution * y_tiles - height) % 2

    full_array = np.pad(array, ((padding_y, padding_y), (padding_x, padding_x)), "constant", constant_values=0)
    full_normalized_array = np.pad(normalized_array, ((padding_y, padding_y), (padding_x, padding_x)), "constant", constant_values=0)

    if not small_units:
        for j in range(y_tiles):
            for i in range(x_tiles):
                left = i * resolution
                upper = j * resolution
                right = left + resolution
                lower = upper + resolution
                if i != x_tiles - 1:
                    for k in range(upper, lower):
                        full_array[k, right - 1:right + 1] = np.mean(full_array[k, right - 2:right + 2])
                        full_normalized_array[k, right - 1:right + 1] = np.mean(full_normalized_array[k, right - 2:right + 2])
                if j != y_tiles - 1:
                    for l in range(left, right):
                        full_array[lower - 1:lower + 1, l] = np.mean(full_array[lower - 2:lower + 2, l])
                        full_normalized_array[lower - 1:lower + 1, l] = np.mean(full_normalized_array[lower - 2:lower + 2, l])

    tiles = []
    for j in range(y_tiles):
        for i in range(x_tiles):
            left = i * resolution
            upper = j * resolution
            right = left + resolution
            lower = upper + resolution
            new_array = full_array[upper:lower, left:right]
            new_normalized_array = full_normalized_array[upper:lower, left:right]

            upper_lat = transform[3] - (upper + padding_y) * transform[5]
            lower_lat = transform[3] - (lower + padding_y) * transform[5]

            if small_units:
                valid_mask = new_array != 0

                if np.any(valid_mask):
                    new_array[valid_mask] = gaussian_filter(new_array[valid_mask], sigma=sigma)

                if np.any(valid_mask):
                    min_val = new_array[valid_mask].min()
                    max_val = new_array[valid_mask].max()
                    if max_val != min_val:
                        with np.errstate(divide="ignore"):
                            normalized_tile = np.where(valid_mask, (new_array - min_val) / (max_val - min_val), 0)
                    else:
                        normalized_tile = np.zeros_like(new_array)
                else:
                    normalized_tile = np.zeros_like(new_array)

                new_img = Image.fromarray((normalized_tile * 65535).astype(np.uint16))
            else:
                new_img = Image.fromarray(new_normalized_array.astype(np.uint16))

            tiles.append((new_img, new_array, upper_lat, lower_lat))
    return tiles, x_tiles, y_tiles
