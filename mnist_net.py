import time
import sqlite3
import pickle

import numpy as np

from mnist_reader import read_file
from neural_net import Neural_Network as NN

DB_FILEPATH = "mnist_data/mnist_data.db"
HIDDEN_SIZE = 28
INPUT_SIZE = 784
OUTPUT_SIZE = 10

TRAINING_SIZE = 60_000
VALIDATION_SIZE = 10_000
TESTING_SIZE = 10_000
BATCH_SIZE = 100

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

def serialize(data_gen):
    # Yield a tuple of pickled data given a generator of data.
    return map(lambda x: (pickle.dumps(x),), data_gen)

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

def create_network(layer_sizes=(2,2,1), activation="relu"):
    # Return a Neural_Network instance with the hyperparameters given for layer sizes (tuple)
    # and activation function (string).
    return NN(layer_sizes[0], layer_sizes[1], layer_sizes[2], activation)

def get_shaped_array(data_tuple, shape_tuple):
    # Return a numpy Array given the tuple data_tuple in the shape given the tuple shape_tuple.
    return np.array(data_tuple).reshape(shape_tuple)

def deserialize(pickled_data_list):
    # Return a tuple of the data unpickled given the pickled data list of byte strings.
    return tuple(map(lambda x: pickle.loads(x[0]), pickled_data_list))

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

def store_data(db_filename, table, total_size, batch_size):
    # Read from the mnist data file and store it to the database by given string db_filename and string table.
    # The int total_size determines how many rows to read and store,
    # and the int batch_size is how much to read into memory at a time.
    for offset in range(0, total_size, batch_size):
        # Put the data into a generator to save memory space.
        # Each iteration in the generator must be a singleton for the SQL to work.
        data_gen = read_file("mnist_data/{}".format(table), batch_size, offset)
        write_to_db(db_filename, table, data_gen)

def convert_int_to_10D(integer):
    vector_10D = np.zeros(10)
    vector_10D[integer] = 1
    return vector_10D

def convert_10D_to_int(vector_10D):
    return list(vector_10D).index(max(vector_10D))


if __name__ == "__main__":
    create_table(DB_FILEPATH)
    for table, total_size in DATA_FILES:
        store_data(DB_FILEPATH, table, total_size, BATCH_SIZE)

    start_time = time.perf_counter()
    x = read_file("mnist_data/training_examples", TRAINING_SIZE, 0)
    y = read_file("mnist_data/training_labels", TRAINING_SIZE, 0)
    print(f"Training data read in {time.perf_counter()-start_time} seconds")
    start_time = time.perf_counter()
    testing_x = read_file("mnist_data/testing_examples", TESTING_SIZE, 0)
    testing_y = read_file("mnist_data/testing_labels", TESTING_SIZE, 0)
    print(f"Testing data read in {time.perf_counter()-start_time} seconds")
    start_time = time.perf_counter()
    # The input vectors must scaled to be between 0 and 1.
    x = np.array(x) / 255
    testing_x = np.array(testing_x) / 255
    # The labels must be converted to 1-hot encoding.
    y = np.array([convert_int_to_10D(i) for i in y])
    testing_y = np.array([convert_int_to_10D(i) for i in testing_y])
    print(f"Data preprocessed in {time.perf_counter()-start_time} seconds")

    nn = create_network((INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE), activation="relu")

    # Test the untrained model as a baseline.
    result = nn.test(x, y)
    print(f"Testing: {result}")
    # Make a copy of the untrained weights if needed later.
    W = nn.W1.copy(), nn.W2.copy()

    def train():
        nn.train(x, y, batch_size=BATCH_SIZE, validation_size=VALIDATION_SIZE, learning_rate=1e-6, epochs=1_000, verbose=True)
    train()
