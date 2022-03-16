import numpy as np
import pandas as pd
import os

BLISS_DIR           = "./"
DEFAULT_DATA_DIR    = "data/"

def read_gaussian_data(bliss_dir = BLISS_DIR, data_dir = DEFAULT_DATA_DIR):
    bliss_N = 512
    type_names = ['seed', 'p1', 'p2', 'num_randombytes_calls', 'utop', 'ubot', 't', 'c', 'z_before', 'Kx', 'yu', 'z']
    dt = np.dtype([
        ('seed', np.ubyte, (16,)),
        ('p1', np.int32, (bliss_N,)),
        ('p2', np.int32, (bliss_N,)),
        ('num_randombytes_calls', np.int32),
        ('utop', np.uint64, (2*bliss_N,)),
        ('ubot', np.uint64, (2*bliss_N,)),
        ('t', np.uint32, (2*bliss_N,)),
        ('c', np.uint32, (2*bliss_N,)),
        ('z_before', np.int32, (2*bliss_N,)),
        ('Kx', np.uint32, (2*bliss_N,)),
        ('yu', np.uint32, (2*bliss_N,)),
        ('z', np.int32, (2*bliss_N,))
    ])
    gaussian_data_file = os.path.join(bliss_dir, data_dir, "data_gaussian.bin")
    data = np.fromfile(gaussian_data_file, dtype=dt)
    data = data.tolist()
    return pd.DataFrame(data, columns=type_names)

def read_bliss_data(bliss_dir = BLISS_DIR, data_dir = DEFAULT_DATA_DIR, attack3 = False):
    bliss_N = 512
    bliss_kappa = 23
    bliss_CRYPTO_SECRETKEYBYTES =  1536
    bliss_MLEN = 512
    if attack3:
        type_names = ['sk', 's1', 's2', 'm','c', 's1c', 's2c', 'b', 'z1', 'z2']
        dt = np.dtype([
            ('sk', np.byte, (bliss_CRYPTO_SECRETKEYBYTES,)),
            ('s1', np.int32, (bliss_N,)),
            ('s2', np.int32, (bliss_N,)),
            ('m', np.byte, (bliss_MLEN,)),
            ('c', np.uint16, (bliss_kappa,)),
            ('s1c', np.int32, (bliss_N,)),
            ('s2c', np.int32, (bliss_N,)),
            ('b', np.byte),
            ('z1', np.int32, (bliss_N,)),
            ('z2', np.int32, (bliss_N,))
        ])
    else:
        type_names = ['sk', 's1', 's2', 'm', 'seed', 'Kx', 'yu', 'z', 'c', 'C_matrix', 'num_attempts', 's1c', 's2c', 'b', 'z1', 'z2']
        dt = np.dtype([
            ('sk', np.byte, (bliss_CRYPTO_SECRETKEYBYTES,)),
            ('s1', np.int32, (bliss_N,)),
            ('s2', np.int32, (bliss_N,)),
            ('m', np.byte, (bliss_MLEN,)),
            ('seed', np.int32, (4,)),
            ('Kx', np.int32, (1024,)),
            ('yu', np.int32, (1024,)),
            ('z', np.int32, (1024,)),
            ('c', np.uint16, (bliss_kappa,)),
            ('C_matrix', np.int32, (bliss_N,bliss_N)),
            ('num_attempts', np.int32),
            ('s1c', np.int32, (bliss_N,)),
            ('s2c', np.int32, (bliss_N,)),
            ('b', np.byte),
            ('z1', np.int32, (bliss_N,)),
            ('z2', np.int32, (bliss_N,))
        ])
    bliss_data_file = os.path.join(bliss_dir, data_dir, "data_attack_bliss.bin")
    data = np.fromfile(bliss_data_file, dtype=dt)
    data = data.tolist()
    return pd.DataFrame(data, columns=type_names)
