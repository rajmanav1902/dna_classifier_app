import numpy as np

# One-hot encoding for DNA
def one_hot_encode(seq, max_len=1000):
    base_map = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1]}
    seq = seq.upper().replace('\n', '').replace(' ', '')
    seq = seq[:max_len].ljust(max_len, 'A')
    return np.array([base_map.get(base, [0, 0, 0, 0]) for base in seq])
