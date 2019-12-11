import json
import numpy as np


# Use read_json to read
def write_player_dict(pd, filename):
    with open(filename, "w") as file:
        # numpy cant be JSONed, so convert
        for key1 in pd:
            for key2 in pd[key1]:
                if pd[key1][key2] is not None:
                    pd[key1][key2] = pd[key1][key2].tolist()
        file.write(json.dumps(pd))
    print("file written: " + str(filename))


def write_np_arr(arr, filename):
    with open(filename, "wb") as np_file:
        np.save(np_file, arr)
    print("file written: " + str(filename))


def write_json(o, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(o))
    print("file written: " + str(filename))


def read_json(filename):
    with open(filename) as f:
        pd = json.load(f)
    return pd


def read_numpy_arr(filename):
    arr = np.load(filename, allow_pickle=True)
    return arr
