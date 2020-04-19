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

def _read_labels(byte_string, chunks, offset):
    # Return a list of ints (0 through 9) given a byte string, an int for chunks and another int for offset.
    end_index = offset + chunks
    return [_to_int(byte_string[i]) for i in range(offset, end_index)]

def _read_images(byte_string, chunks, offset):
    # Return a chunks long list of list of ints given a byte string and an int chunks and an int offset.
    rows = _to_int(byte_string[:4])
    columns = _to_int(byte_string[4:8])
    # The rows and columns of the images should be 28.
    if rows != 28 or columns != 28:
        return ValueError
    image_size = rows * columns
    start_pixel = offset * image_size
    end_pixel = start_pixel + image_size * chunks

    # Remove the first 8 bytes as they are only meta data for the dimensions.
    byte_string = byte_string[8:]
    # Convert the necessary number of pixel bytes into ints.
    int_list = [byte_string[i] for i in range(start_pixel, end_pixel)]
    flattened_images = [int_list[image_size*i:image_size*(i+1)] for i in range(chunks)]
    return flattened_images


def read_file(filename, chunks=None, offset=0):
    """Return a list or nested list given a filename for the MNIST data,
    an int for how many items to return as well as the offset of where to start reading.
    """
    with open(filename, "rb") as file:
        contents = file.read()
        meta_data, data = contents[:8], contents[8:]
    magic_byte = _to_int(meta_data[:4])
    if chunks is None:
        chunks = _to_int(meta_data[4:8])
    # This is for the labels.
    if magic_byte == 2049:
        data_list = _read_labels(data, chunks, offset)
    # This is for the images.
    elif magic_byte == 2051:
        data_list = _read_images(data, chunks, offset)
    # An error should be raised if attempting to read in the wrong file.
    else:
        return KeyError
    return data_list
