import time
import sqlite3
import pickle

import numpy as np

from mnist_reader import read_file
from neural_net import Neural_Network as NN

DB_FILEPATH = "mnist_data/mnist_data.db"
HIDDEN_SIZE = 400
INPUT_SIZE = 28**2
OUTPUT_SIZE = 10

TRAINING_SIZE = 60_000
TESTING_SIZE = 10_000
BATCH_SIZE = 1_000

DATA_FILES = (
            ("testing_labels", TESTING_SIZE),
            ("training_labels", TRAINING_SIZE),
            ("testing_examples", TESTING_SIZE),
            ("training_examples", TRAINING_SIZE),
)


def log(function):
    # Provide logging functionality.
    def wrapper(*args, **kwargs):
        name = function.__name__
        print(f"Starting {name}({args})")
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        stop_time = time.perf_counter()
        print(f"{name}() completed in {round(stop_time-start_time, 3)} second(s)")
        return result
    return wrapper

@log
def create_table(filename):
    # Create a sqlite database given a filename string.
    conn = sqlite3.connect(filename)
    try:
        with conn as c:
            c.execute("""CREATE TABLE training_examples
                    (data BLOB NOT NULL)""")
        with conn as c:
            c.execute("""CREATE TABLE testing_examples
                    (data BLOB NOT NULL)""")
        with conn as c:
            c.execute("""CREATE TABLE training_labels
                    (data BLOB NOT NULL)""")
        with conn as c:
            c.execute("""CREATE TABLE testing_labels
                    (data BLOB NOT NULL)""")
    except sqlite3.OperationalError as e:
        print(e)
    finally:
        conn.close()

@log
def serialize(data_gen):
    # Yield a tuple of pickled data given a generator of data.
    return map(lambda x: (pickle.dumps(x),), data_gen)

@log
def write_to_db(filename, table, data_gen):
    # Write the items of a given generator of data, given a filename string and table name string.
    # data must be an iterable.
    try:
        conn = sqlite3.connect(filename)
        with conn as c:
            # executemany expects a iterable of tuples.
            c.executemany("""INSERT INTO {} VALUES (?)""".format(table), serialize(data_gen))
    except sqlite3.OperationalError as e:
        print(e)
    finally:
        c.close()

@log
def create_network(layer_sizes=(2,2,1), activation="sigmoid"):
    # Return a Neural_Network instance with the hyperparameters given for layer sizes (tuple)
    # and activation function (string).
    return NN(layer_sizes[0], layer_sizes[1], layer_sizes[2], activation)

@log
def get_shaped_array(data_tuple, shape_tuple):
    # Return a numpy Array given the tuple data_tuple in the shape given the tuple shape_tuple.
    return np.array(data_tuple).reshape(shape_tuple)

@log
def deserialize(pickled_data_list):
    # Return a tuple of the data unpickled given the pickled data list of byte strings.
    return tuple(map(lambda x: pickle.loads(x[0]), pickled_data_list))

@log
def retrieve_data(db_filename, table, offset, limit):
    # Return a tuple of the data from the database specified by the string db_filename and the string table
    # starting at row given by the int offset. The given int limit determines the amount of rows to return.
    try:
        conn = sqlite3.connect(db_filename)
        with conn as c:
            query = c.execute("SELECT data FROM {} WHERE ROWID>={} LIMIT {}".format(table, offset, limit)).fetchall()
    except sqlite3.OperationalError as e:
        print(e)
    else:
        return deserialize(query)
    finally:
        c.close()

@log
def store_data(db_filename, table, total_size, batch_size):
    # Read from the mnist data file and store it to the database by given string db_filename and string table.
    # The int total_size determines how many rows to read and store,
    # and the int batch_size is how much to read into memory at a time.
    for offset in range(0, total_size, batch_size):
        # Put the data into a generator to save memory space.
        # Each iteration in the generator must be a singleton for the SQL to work.
        data_gen = read_file("mnist_data/{}".format(table), batch_size, offset)
        write_to_db(db_filename, table, data_gen)

def convert_1D_to_10D(vector_1D):
    i = vector_1D[0]
    vector_10D = np.zeros(10)
    vector_10D[i] = 1
    return vector_10D

def convert_10D_to_int(vector_10D):
    return list(vector_10D).index(max(vector_10D))


if __name__ == "__main__":
    create_table(DB_FILEPATH)
    for table, total_size in DATA_FILES:
        store_data(DB_FILEPATH, table, total_size, BATCH_SIZE)
    testing_y = get_shaped_array(
            retrieve_data(DB_FILEPATH, "testing_labels", 0, BATCH_SIZE),
            (BATCH_SIZE, 1)
    )
    testing_x = get_shaped_array(
            retrieve_data(DB_FILEPATH, "testing_examples", 0, BATCH_SIZE),
            (BATCH_SIZE, 28, 28)
    )
    # The input matrices must be flattened to vectors and scaled to be between 0 and 1.
    x = np.array([np.ravel(x) for x in testing_x]) / 255
    y = np.array([convert_1D_to_10D(y) for y in testing_y])

    nn = create_network((INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE), activation="relu")
