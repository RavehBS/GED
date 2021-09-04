"""Triplet sampling utils."""

import numpy as np
from bisect import bisect_left,bisect
from tqdm import tqdm



def samples_triples(n_nodes, num_samples):
    num_samples = int(num_samples)
    all_nodes = np.arange(n_nodes)
    mesh = np.array(np.meshgrid(all_nodes, all_nodes))
    pairs = mesh.T.reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    n_pairs = pairs.shape[0]
    if num_samples < n_pairs:
        print("Generating all pairs subset")
        subset = np.random.choice(np.arange(n_pairs), num_samples, replace=False)
        pairs = pairs[subset]
    else:
        print("Generating all pairs superset")
        k_base = int(num_samples / n_pairs)
        k_rem = num_samples - (k_base * n_pairs)
        subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
        pairs_rem = pairs[subset]
        pairs_base = np.repeat(np.expand_dims(pairs, 0), k_base, axis=0).reshape((-1, 2))
        pairs = np.concatenate([pairs_base, pairs_rem], axis=0)
    num_samples = pairs.shape[0]
    #TODO: note that this code doesn't avoid taking the same triples: [1,2,3]==[3,2,1] and bad triples, e.g. [1,2,1]
    triples = np.concatenate(
        [pairs, np.random.randint(n_nodes, size=(num_samples, 1))],
        axis=1
    )
    return triples


def generate_all_triples(n_nodes):
    triples = []
    for n1 in tqdm(np.arange(n_nodes)):
        for n2 in np.arange(n1 + 1, n_nodes):
            for n3 in np.arange(n2 + 1, n_nodes):
                triples += [(n1, n2, n3)]
    return np.array(triples)

def find_triplet_at_idx(idx,n_nodes):
    cnt = 0
    for n1 in np.arange(n_nodes):
        for n2 in np.arange(n1+1, n_nodes):
            for n3 in np.arange(n2+1,n_nodes):
                if cnt == idx:
                    return (n1, n2 ,n3)
                cnt += 1


def init_sum_of_triangular_series(n):
    sk = [np.uint(i*(i+1)*(i+2)/6) for i in range(n-1)]
    return sk

def init_triangular_series(n):
    sk = [np.uint(i*(i+1)/2) for i in range(n-1)]
    return sk


def find_1st_bit(idx,n,SK):
    max_1st_bit = n-3
    if idx == 0:
        return max_1st_bit,SK[0]

    idx_found = bisect(SK,idx,0,max_1st_bit+1)-1
    digit = max_1st_bit-idx_found
    diff = SK[idx_found]
    return digit, diff


def find_2nd_bit(rel_idx, bit1, n,SK):
    max_2nd_bit = n - 2
    if rel_idx == 0:
        return max_2nd_bit, SK[0]

    idx_found = bisect(SK,rel_idx,0,max_2nd_bit+1)-1
    digit = max_2nd_bit-idx_found
    diff = SK[idx_found]
    return digit, diff

def find_3rd_bit(rel_idx,bit2,n):
    max_1st_bit = n - 1
    return int(max_1st_bit - rel_idx)

def find_triplet_by_idx(idx,n):
    sk1=init_sum_of_triangular_series(n)
    assert(sk1[-1] > idx) #check idx required is valid
    sk2=init_triangular_series(n)
    bit1,diff1 = find_1st_bit(idx,n,sk1)
    bit2,diff2 = find_2nd_bit(idx-diff1,bit1,n,sk2)
    bit3 = find_3rd_bit(idx-diff1-diff2,bit2,n)
    return np.array([bit1,bit2,bit3])

if __name__ == "__main__":
    N = 20
    NODES = 6
    stack = []
    for i in range(N):
        val = find_triplet_by_idx(i,NODES)
        print(val)
        stack.append(val)

    print("Comparing...")
    for i in range(N):
        a = find_triplet_at_idx(i,NODES)
        b = stack.pop()
        if a == b :
            print("good")#nothing
        else:
            print(f"a = {a} || b = {b}")
    print("Done")
    #a = [0,1,3,7,8,20]
    #k = bisect_left(a,7)
    #print(k)



