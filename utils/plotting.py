import h5py
import numpy as np


def create_data_structure(mode, approach, date):

    filename = "outdata/{m:s}/{a:s}/{d:s}.hdf5".format(m=mode, a=approach, d=date)
    try:
        h5py.File(filename, 'r')
    except OSError:
        with h5py.File(filename, 'a') as f:
            f.create_group("objective")
            if mode == "Tohoku":
                ggroup = f.create_group("gauge_data")
                tvgroup = f.create_group("total_variation")
                ggroup.create_group("P02")
                ggroup.create_group("P06")
                tvgroup.create_group("TVP02")
                tvgroup.create_group("TVP06")


def store_data(mode, approach, date, index, quantities):

    index_str = str(index)
    filename = "outdata/{m:s}/{a:s}/{d:s}.hdf5".format(m=mode, a=approach, d=date)
    with h5py.File(filename, 'a') as f:
        f["objective"].create_dataset(index_str, data=np.array(quantities["J_h"]))

        if mode == "Tohoku":
            f["gauge_data"]["P02"].create_dataset(index_str, data=np.array(quantities["P02"]))
            f["gauge_data"]["P06"].create_dataset(index_str, data=np.array(quantities["P06"]))
            f["total_variation"]["TVP02"].create_dataset(index_str, data=np.array(quantities["TV P06"]))
            f["total_variation"]["TVP06"].create_dataset(index_str, data=np.array(quantities["TV P06"]))


if __name__ == "__main__":

    # TESTING
    quantities = {"J_h": 1., "P02": [0., 1.], "P06": [1., 0.], "TV P02": 0., "TV P06": -1.}
    create_data_structure('Tohoku', 'FixedMesh', 'test')
    store_data('Tohoku', 'FixedMesh', 'test', 0, quantities)
