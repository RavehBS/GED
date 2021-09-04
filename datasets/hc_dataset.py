"""Hierarchical clustering dataset."""

import logging

import numpy as np
import torch
import torch.utils.data as data
from math import comb

import datasets.triples as triplets


class HCDataset(data.Dataset):
    """Hierarchical clustering dataset."""

    def __init__(self, features, labels, similarities, num_samples,generate_triplets_on_the_fly = False):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        @param similarities: pairwise similarities between datapoints
        @type similarities: np.array of shape (n_datapoints, n_datapoints)
        """
        self.features = features
        self.labels = labels
        self.similarities = similarities
        self.n_nodes = self.similarities.shape[0]
        self.num_samples = num_samples
        self.generate_triplets_on_the_fly = generate_triplets_on_the_fly
        if generate_triplets_on_the_fly:
            assert (num_samples < 2) #generating on the fly is possible only when generating all the possible triplets.
            #SK is series of sum of traingular series.
            self.sum_of_triangular_series = triplets.init_sum_of_triangular_series(self.n_nodes)
            self.triangular_series = triplets.ini_triangular_series(self.n_nodes)
            self.triples = None
        else:
            self.triples = self.generate_triples(num_samples)

    def __len__(self):
        if self.generate_triplets_on_the_fly:
            return comb(self.n_nodes, 3)
        else:
            return len(self.triples)

    def __getitem__(self, idx):
        if self.generate_triplets_on_the_fly:
            triple = triplets.find_triplet_by_idx(idx, self.n_nodes)
        else:
            triple = self.triples[idx]
        s12 = self.similarities[triple[0], triple[1]]
        s13 = self.similarities[triple[0], triple[2]]
        s23 = self.similarities[triple[1], triple[2]]
        similarities = np.array([s12, s13, s23])
        return torch.from_numpy(triple), torch.from_numpy(similarities)

    def generate_triples(self, num_samples):
        logging.info("Generating triples.")
        assert (self.generate_triplets_on_the_fly is False)
        if num_samples < 0:
            triples = triplets.generate_all_triples(self.n_nodes)
        else:
            triples = triplets.samples_triples(self.n_nodes, num_samples=num_samples)
        logging.info(f"Total of {triples.shape[0]} triples")
        return triples.astype("int64")