

"""Dataset loading."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
import scipy.io
import random

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]
ZOO_DICT = ["Mammals","Birds", "Reptiles","Fish", "Amphibia","Insects", "Shellfish"]

HYPBC_DATASETS = [
    "bc"
]

def load_data(dataset, normalize=True, data_size = None):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param data_size: number of samples in dataset
    @type data_size: int
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset,data_size)
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
    ####DEBUG####
    matrix_histogram(similarities)

    dict = ZOO_DICT if dataset == "zoo" else None
    return x, y, similarities, dict


def load_uci_data(dataset,data_size = None):
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
    i = 0
    with open(data_path, 'r') as f:
        for line in f:
            i = i + 1
            split_line = line.split(",")

            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
            if data_size is not None and i == data_size:
                break
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    epslion = 1e-2
    std = x.std(0)
    x = (x - mean) / (std + epslion)

    return x, y


def get_d2(data, label, os_time, os_sta, patient_list):
    returned_data  = []
    returned_label = []
    returned_os_time = []
    returned_os_stat = []
    shape = patient_list.shape[0]

    label_dict = ['LumA','LumB','Basal','Her2','Normal']

    for i in range(shape):
        if label[i].size > 0 and patient_list[i]:
            if label[i][0] == 'LumA':
                returned_label.append(0)
                returned_data.append(data[:,i])
                returned_os_time.append(os_time[i][0].astype(float))
                if os_sta[i][0] == 'LIVING':
                    returned_os_stat.append(0)
                else:
                    returned_os_stat.append(1)
            elif label[i][0] == 'LumB':
                returned_label.append(1)
                returned_data.append(data[:,i])
                returned_os_time.append(os_time[i][0].astype(float))
                if os_sta[i][0] == 'LIVING':
                    returned_os_stat.append(0)
                else:
                    returned_os_stat.append(1)
            elif label[i][0] == 'Basal':
                returned_label.append(2)
                returned_data.append(data[:,i])
                returned_os_time.append(os_time[i][0].astype(float))
                if os_sta[i][0] == 'LIVING':
                    returned_os_stat.append(0)
                else:
                    returned_os_stat.append(1)
            elif label[i][0] == 'Her2':
                returned_label.append(3)
                returned_data.append(data[:,i])
                returned_os_time.append(os_time[i][0].astype(float))
                if os_sta[i][0] == 'LIVING':
                    returned_os_stat.append(0)
                else:
                    returned_os_stat.append(1)
            elif label[i][0] == 'Normal':
                returned_label.append(4)
                returned_data.append(data[:,i])
                returned_os_time.append(os_time[i][0].astype(float))
                if os_sta[i][0] == 'LIVING':
                    returned_os_stat.append(0)
                else:
                    returned_os_stat.append(1)
    return returned_data, returned_os_time, returned_os_stat, returned_label, label_dict

def load_hypbc_5_type_metabric(visualize = False, corr_thresh = 0.9,num_features=-1):
    data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/dataStructMetabric.mat")

    mat2 = scipy.io.loadmat(data_path)
    gene_data2 = mat2['dataStructMetabric']['data'][0][0]
    #remove nan
    gene_data2[np.where(np.isnan(gene_data2))[0], np.where(np.isnan(gene_data2))[1]] = 0

    patient_list2 = np.sum(gene_data2, axis=0) != 0

    clinic2_info = mat2['dataStructMetabric']['clinic']
    clinVariable2 = clinic2_info[0, 0]['clinVariable'][0][0]
    clinData2 = clinic2_info[0, 0]['data'][0][0]
    geneList2 = mat2['dataStructMetabric']['genes'][0][0]

    data2, os_time2, os_stat2, label2,label_dict = get_d2(gene_data2, clinData2[:, 6], clinData2[:, 22], clinData2[:, 23],
                                               patient_list2)
    data2 = np.vstack(data2) #gene expression data, each row is patient
    label2 = np.asarray(label2) #matching cancer types
    sample_names = [] #patient specific ID
    feat_names = [] #gene names


    data2,label2,geneList2,sample_names, feat_names, label_dict = \
        preprocess_data(data2,label2,geneList2,sample_names, feat_names, label_dict,
                    num_features = num_features,filter_by_corr = True,corr_thresh = corr_thresh)

    return data2, label2, sample_names, feat_names, label_dict

def preprocess_data(data,labels,geneList,sample_names, feat_names, label_dict,
                    num_features = -1,filter_by_corr = True,corr_thresh = 0.9):

    data2 = data
    labels2 = labels
    geneList2 = geneList

    #sort data  by feature variance
    feature_variance = np.var(data2,axis=0)
    sorted_ind = np.flipud(np.argsort(feature_variance)) #sort in descending manner
    sorted_data_by_variance = data2[:,sorted_ind]
    geneList2 = geneList2[sorted_ind]

    # remove features with constant values
    sorted_feature_variance = feature_variance[sorted_ind]
    non_const_features_ind = np.where(sorted_feature_variance > 0)
    sorted_data_by_variance = np.squeeze(sorted_data_by_variance[:,non_const_features_ind])
    geneList2 = geneList2[non_const_features_ind]

    if num_features > 1:
        sorted_data_by_variance = sorted_data_by_variance[:,:num_features]
        geneList2 = geneList2[:num_features]

    if filter_by_corr:
        # remove correlated features
        feature_correlation = np.corrcoef(x=sorted_data_by_variance,rowvar=False)
        col_to_drop = []
        n = len(feature_correlation)
        for i in range(n):
            for j in range(i+1,n): #only check triangle above the diagonal
                if i not in col_to_drop:
                    if np.abs(feature_correlation[i,j]) > corr_thresh:
                        col_to_drop.append(j)
        bool_col_to_drop = np.ones(n,dtype=bool)
        bool_col_to_drop[col_to_drop] = False
        print(bool_col_to_drop)
        filtered_data = sorted_data_by_variance[:,bool_col_to_drop]
        geneList2 = geneList2[bool_col_to_drop]

    return filtered_data, labels2, geneList2,sample_names, feat_names, label_dict



def load_hypbc_data_deprecated(type="all", normalize ="none", visualize=False):
    if type == "all":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna_all.txt")
    elif type == "partial":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna.txt")
    else:
        print("bad type, use: \"partial\" or \"all\"")
        return
    clinical_data_path = os.path.join(os.environ["DATAPATH"],"breast_cancer/clinical_data.txt")

    #read data csv
    df = pd.read_csv(data_path, sep="\t", header=0)
    df.dropna(axis=0, inplace=True, how="any")
    df.drop(axis=1, inplace=True, labels="Entrez_Gene_Id")

    #sort features by variance
    df["variance"] = df.var(axis=1, numeric_only=True)
    df.sort_values(by=["variance"], ascending=False, inplace=True)

    if visualize:
        #add more information

        df["min"] = df.min(axis=1, numeric_only=True)
        df["max"] = df.max(axis=1, numeric_only=True)
        df["mean"] = df.mean(axis=1, numeric_only=True)


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

        df.drop(labels=["min","max","mean"],inplace=True,axis=1)


    #load clinical data
    clinical_df = pd.read_csv(clinical_data_path, sep="\t", header=4)

    # add the clinical data to the samples
    transposed_df = df.transpose()
    transposed_df.reset_index(inplace=True)
    transposed_df.rename(columns={"index": "SAMPLE_ID"}, inplace=True)
    df_full_data = pd.merge(transposed_df, clinical_df, on="SAMPLE_ID") #df_full: rows are patients, colums are features + clinical data

    #generate labels
    labels = df_full_data["CANCER_TYPE_DETAILED"]
    label_dict = {}
    for i,val in enumerate(labels.unique().tolist()):
        label_dict[val] = i

    labels.replace(label_dict, inplace=True)


    #generate the data
    hugo_sym = df["Hugo_Symbol"].to_list()
    df.drop(labels=['Hugo_Symbol','variance'], inplace=True,axis=1)
    sample_names = df.columns.values

    return df.to_numpy().transpose(), labels.to_numpy(), hugo_sym, sample_names, list(label_dict.keys())

def generate_similarity_matrix(data, method = 'euclidean', features_dim = 1000,visualize=False):
    #assume data is a dataframe where each row is a sample and each column  is a feature
    filt_data = data[:,:features_dim]
    dist = sp.distance.cdist(filt_data,filt_data,method)

    #normalize similarity to [0,1]
    if method == 'cosine':
        mat = 0.5*(2-dist) # note that dist = 1-cos(t) => mat = 0.5(1 + cos(t))
    else:
        mat = 1-dist*(1/(np.amax(dist)))

    if visualize:
        v = plt.imshow(mat,cmap="gray")
        plt.title(f"similarity matrix, method:{method},#dim={features_dim}")
        plt.show()

    return mat

def split_data_to_groups(data, labels, n_groups):
    rng = np.random.default_rng(0)  # for the same split everytime
    possible_labels = np.sort(np.unique(labels))
    data_indices_per_label = []

    for l in possible_labels:
        indices = np.array(np.where(labels == l))[0]
        rng.shuffle(indices)
        data_indices_per_label.append(np.array_split(indices, n_groups))
    #{[()()],
    # [()()]
    # [()()]}

    data_indices_per_label = np.array(data_indices_per_label)
    #aggregate requested group
    all_groups = []
    for i in range(data_indices_per_label.shape[1]):#for each group
        selected_group = []
        for j in range(data_indices_per_label.shape[0]): #for each label
            selected_group = selected_group + data_indices_per_label[j][i].tolist()
        all_groups.append(selected_group)

    all_data_groups = [data[g] for g in all_groups]
    all_label_groups = [labels[g] for g in all_groups]
    return all_data_groups, all_label_groups

def choose_one_from_each(data,labels,num_data_samples):
    #choose num_data_samples patients from data.
    unique_labels = np.unique(labels).tolist()

    if num_data_samples > len(unique_labels):
        indices_chosen = []
        #take at least 1 of every label.
        indices = np.arange(len(labels))
        mask = np.ones_like(indices,dtype=bool)
        for label in unique_labels:
            cur_label_indices = np.where(labels==label)
            chosen_idx = np.random.choice(cur_label_indices[0])
            mask[chosen_idx] = False
            indices_chosen.append(chosen_idx)


        #indices left_after_choosing_one_for_each_label
        indices = indices[mask]

        #choose the rest of the samples.
        chosen_indices_of_the_indices = np.random.choice(len(indices), num_data_samples - len(unique_labels))
        leftover_indices_chosen = indices[chosen_indices_of_the_indices]
        indices_chosen = np.concatenate([indices_chosen,leftover_indices_chosen])

        assert(len(indices_chosen) == num_data_samples)
        data_out = data[indices_chosen]
        labels_out = labels[indices_chosen]

        return data_out,labels_out



def load_hypbc(type="partial",
               normalize = "none",
               num_data_samples = -1,
               feature_dim = 50,
               method = "cosine",
               feature_correlation_thresh = 0.9,
               visualize=False):
    data, labels, feat_names, samp_names, label_dict = load_hypbc_5_type_metabric(num_features=feature_dim, corr_thresh=feature_correlation_thresh,visualize=visualize)
    # data, labels, feat_names, samp_names, label_dict = load_hypbc_data_deprecated(type=type,
    #                                                                               normalize=normalize,
    #                                                                               visualize=visualize)
    data,labels = choose_one_from_each(data,labels,num_data_samples)

    sim_mat = generate_similarity_matrix(data,
                                         features_dim=feature_dim,
                                         method=method,
                                         visualize=visualize)

    return data, labels, sim_mat, label_dict

def load_hypbc_multi_group(num_data_samples = -1,
               num_groups = 1,
               group_idx = 0,
               feature_dim = 50,
               method = "cosine",
               feature_correlation_thresh = 0.9,
               visualize=False):
    data, labels, feat_names, samp_names, label_dict = load_hypbc_5_type_metabric(num_features=feature_dim,
                                                                                  corr_thresh=feature_correlation_thresh,
                                                                                  visualize=visualize)

    # decide on data group size
    n = len(labels) if num_data_samples == -1 else num_data_samples
    min_num_of_groups = np.ceil(len(labels)/n)
    num_groups = max(num_groups, min_num_of_groups)
    assert (num_groups > group_idx)

    #split data
    data, labels = split_data_to_groups(data,labels,num_groups)

    sim_matrices = [generate_similarity_matrix(data[i],
                                         features_dim=feature_dim,
                                         method=method,
                                         visualize=visualize) for i in range(len(data))]

    return data, labels, sim_matrices, label_dict



def matrix_histogram(matrix):
    vector = matrix.flatten()
    #hist, bin_edges = np.histogram(vector)
    plt.figure()
    _ = plt.hist(vector,bins='auto')
    plt.show()

if __name__ == "__main__":
    load_hypbc()

