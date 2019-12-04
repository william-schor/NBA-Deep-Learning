import json
import numpy as np


def write_dict(pd, filename):
    with open(filename, "w") as file:
        # numpy cant be JSONed, so convert
        for key1 in pd:
            for key2 in pd[key1]:
                if pd[key1][key2] is not None:
                    pd[key1][key2] = pd[key1][key2].tolist()
        file.write(json.dumps(pd))
    print(f"file written: {filename}")


def write_np_arr(arr, filename):
    with open(filename, "wb") as np_file:
        np.save(np_file, arr)
    print(f"file written: {filename}")


def read_dict(filename):
    with open(filename) as f:
        pd = json.load(f)
    return pd


def read_numpy_arr(filename):
    arr = np.load("wl_per_rosters.npy", allow_pickle=True)
    return arr
