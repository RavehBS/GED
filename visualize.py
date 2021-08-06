"""Script to visualize the HypHC clustering."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from datasets.loading import load_data
from model.hyphc import HypHC
from utils.poincare import project
from utils.visualization import plot_tree_from_leaves
from datasets.loading import load_hypbc

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=0, help="model seed to use")
    args = parser.parse_args()

    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
    #_, y_true, similarities = load_data(config_args.dataset)
    _, y_true, similarities = load_hypbc(type="partial",
               normalize = "none",
               feature_dim = 3,
               method = config_args.similarity_metric,
               visualize=False)

    # build HypHC model
    model = HypHC(similarities.shape[0], config_args.rank, config_args.temperature, config_args.init_size,
                  config_args.max_scale)
    params = torch.load(os.path.join(args.model_dir, f"model_{args.seed}.pkl"), map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=False)
    model.eval()

    # decode tree
    tree = model.decode_tree(fast_decoding=True)
    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)
    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax = plot_tree_from_leaves(ax, tree, leaves_embeddings, labels=y_true)
    fig.savefig(os.path.join(args.model_dir, f"embeddings_{args.seed}.png"))

def visualize_tree(model,tree,y_true, save_path,label_dict):
    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)
    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax = plot_tree_from_leaves(ax, tree, leaves_embeddings, label_dict=label_dict, labels=y_true)
    fig.savefig(save_path)

def visualize_epoch(model,model_state_dir,y_true,epoch_idx):
    if model_state_dir != '':
        #load model
        params = torch.load(os.path.join(model_state_dir, f"model_{epoch_idx}.pkl"), map_location=torch.device('cpu'))
        model.load_state_dict(params, strict=False)
    model.eval()

    # decode tree
    tree = model.decode_tree(fast_decoding=True)
    save_path = os.path.join(model_state_dir, f"embeddings_{epoch_idx}.png")
    visualize_tree(model,y_true, save_path)


def visualize_training_from_file(args):
    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
    #_, y_true, similarities = load_data(config_args.dataset)
    _, y_true, similarities = load_hypbc(type="partial",
               normalize = "none",
               feature_dim = 3,
               method = config_args.similarity_metric,
               visualize=False)

    # build HypHC model
    model = HypHC(similarities.shape[0], config_args.rank, config_args.temperature, config_args.init_size,
                  config_args.max_scale)

    #TODO: get all the correct file names, extract epoch idx
    files = os.path.dir(args.model_dir)

    for file in files: #TODO: iterate over epoch files and visualize
        visualize_epoch(model,model_state_dir = files,epoch_idx = 0)

def main2():
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=0, help="model seed to use")
    args = parser.parse_args()
    visualize_training_from_file(args)


