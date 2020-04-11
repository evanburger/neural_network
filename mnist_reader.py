"""This module is necessary to read in the data from the MNIST data set file format."""
# import click


BYTE_ORDER = "big"


def _to_int(byte_string, is_signed=False):
    # Return an int given a byte string.
    try:
        return int.from_bytes(byte_string, byteorder=BYTE_ORDER, signed=is_signed)
    # The code won't work if the byte_string is a single character.
    except TypeError:
        return byte_string

def _read_labels(byte_string):
    # Return a list of ints (0 through 9) given a byte string.
    return [_to_int(byte_char) for byte_char in byte_string]

def _read_images(byte_string, image_amount):
    # Return a 28 by 28 by image_amount nested list given a byte string
    # and an int of the amount of images.
    rows = _to_int(byte_string[:4])
    columns = _to_int(byte_string[4:8])
    # The rows and columns of the images should be 28.
    if rows != 28 or columns != 28:
        return ValueError
    # Remove the first 8 bytes as they are only meta data for the dimensions.
    byte_string = byte_string[8:]
    # Convert byte string into a nested list of size (images x rows x columns).
    images = [[byte_string[j:j+columns] for j in range(0, len(byte_string), columns)][i:i+rows] for i in range(0, rows*image_amount, rows)]
    if len(images) != image_amount:
        return ValueError
    return images


def read_file(filename):
    """Return a list or nested list given a filename for the MNIST data."""
    with open(filename, "rb") as file:
        contents = file.read()
        meta_data, data = contents[:8], contents[8:]
    magic_byte = _to_int(meta_data[:4])
    example_size = _to_int(meta_data[4:8])
    # This is for the labels.
    if magic_byte == 2049:
        data_list = _read_labels(data)
    # This is for the images.
    elif magic_byte == 2051:
        data_list = _read_images(data)
    # An error should be raised if attempting to read in the wrong file.
    else:
        return KeyError
    return data_list
