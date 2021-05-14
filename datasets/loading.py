"""Dataset loading."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]

HYPBC_DATASETS = [
    "bc"
]

def load_data(dataset, normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset)
    elif dataset in HYPBC_DATASETS:
        x,y = load_hypbc()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = np.triu(similarities) + np.triu(similarities).T
    similarities[np.diag_indices_from(similarities)] = 1.0
    similarities[similarities > 1.0] = 1.0
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    return x, y

def load_hypbc(type="all",normalize = "none",visualize=False):
    if type == "all":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna_all.txt")
    elif type == "partial":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna.txt")
    else:
        print("bad type, use: \"partial\" or \"all\"")
        return
    #read csv
    df = pd.read_csv(data_path, sep="\t", header=0)
    df.dropna(axis=0, inplace=True, how="any")
    df.drop(axis=1, inplace=True, labels="Entrez_Gene_Id")
    if visualize:
        #add more information
        df["variance"] = df.var(axis = 1, numeric_only=True)
        df["min"] = df.min(axis=1, numeric_only=True)
        df["max"] = df.max(axis=1, numeric_only=True)
        df["mean"] = df.mean(axis=1, numeric_only=True)
        df.sort_values(by=["variance"],ascending=False,inplace=True)

        #see some samples and shape
        print(df.shape)
        print(df.loc[0:5, :])

        #plot
        fig, axes = plt.subplots(nrows=1, ncols=5)
        fig.suptitle("5 most variant genes")
        fig.tight_layout()
        for i in range(5):
            cur_frame = df.iloc[i,:]
            var = cur_frame["variance"]
            mean = cur_frame["mean"]
            cur_frame = cur_frame.drop(labels=["Hugo_Symbol","min","max","mean","variance"])
            axes[i].hist(cur_frame, bins=50)
            axes[i].set_title('mean,var=({:.2f},{:.2f})'.format(mean, var),fontsize=8)

    return df
if __name__ == "__main__":
    data = load_hypbc(type="partial",normalize = "none",visualize=True)