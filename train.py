"""Train a hyperbolic embedding model for hierarchical clustering."""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

import optim
from config import config_args
from datasets.hc_dataset import HCDataset
from datasets.loading import load_data
from datasets.loading import load_hypbc
from datasets.loading import load_hypbc_multi_group
from model.hyphc import HypHC
from utils.metrics import dasgupta_cost
from utils.training import add_flags_from_config, get_savedir
from visualize import visualize_tree



def train_internal(args,x, y_true, similarities, label_dict,prefix=""):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # get saving directory
    if args.save:
        save_dir = get_savedir(args,prefix)
        logging.info("Save directory: " + save_dir)
        save_path = os.path.join(save_dir, "model_{}.pkl".format(args.seed))
        if os.path.exists(save_dir):
            if os.path.exists(save_path):
                logging.info("Model with the same configuration parameters already exists.")
                logging.info("Exiting")
                return
        else:
            os.makedirs(save_dir)
            with open(os.path.join(save_dir, "config.json"), 'w') as fp:
                json.dump(args.__dict__, fp)
        log_path = os.path.join(save_dir, "train_{}.log".format(args.seed))
        hdlr = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    # set seed
    logging.info("Using seed {}.".format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set precision
    logging.info("Using {} precision.".format(args.dtype))
    if args.dtype == "double":
        torch.set_default_dtype(torch.float64)

    dataset = HCDataset(x, y_true, similarities, num_samples=args.num_samples)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # create model
    model = HypHC(dataset.n_nodes, args.rank, args.temperature, args.init_size, args.max_scale)
    model.to("cuda")

    # create optimizer
    Optimizer = getattr(optim, args.optimizer)
    optimizer = Optimizer(model.parameters(), args.learning_rate)

    # train model
    best_cost = np.inf
    best_model = None
    counter = 0
    logging.info("Start training")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(dataloader), unit='ex') as bar:
            for step, (triple_ids, triple_similarities) in enumerate(dataloader):
                #for param in model.parameters():
                    #print(param.data)
                #TODO: generate triplets on the fly
                triple_ids = triple_ids.cuda()
                triple_similarities = triple_similarities.cuda()
                loss = model.loss(triple_ids, triple_similarities)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.6f}')
                total_loss += loss
        total_loss = total_loss.item() / (step + 1.0)
        logging.info("\t Epoch {} | average train loss: {:.6f}".format(epoch, total_loss))

        # keep best embeddings
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            tree = model.decode_tree(fast_decoding=args.fast_decoding)

            #save embedding and weights for this epoch
            model_path = os.path.join(save_dir, f"model_sd{args.seed}_epch{epoch}.pkl")
            torch.save(model.state_dict(), model_path)
            img_path = os.path.join(save_dir, f"embedding_sd{args.seed}_epch{epoch}.png")
            visualize_tree(model, tree, y_true, img_path,label_dict)

            cost = dasgupta_cost(tree, similarities)
            logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))
            if cost < best_cost:
                counter = 0
                best_cost = cost
                best_model = model.state_dict()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("Early stopping.")
                    break

        # anneal temperature
        if (epoch + 1) % args.anneal_every == 0:
            model.anneal_temperature(args.anneal_factor)
            logging.info("Annealing temperature to: {}".format(model.temperature))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_factor
                lr = param_group['lr']
            logging.info("Annealing learning rate to: {}".format(lr))

    logging.info("Optimization finished.")
    if best_model is not None:
        # load best model
        model.load_state_dict(best_model)

    if args.save:
        # save best embeddings
        logging.info("Saving best model at {}".format(save_path))
        torch.save(best_model, save_path)

    # evaluation
    model.eval()
    logging.info("Decoding embeddings.")
    tree = model.decode_tree(fast_decoding=args.fast_decoding)
    cost = dasgupta_cost(tree, similarities)
    logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))

    if args.save:
        logger.removeHandler(hdlr)
    return

def multi_train(args,num_groups):
    # create dataset
    label_dict = None
    if args.dataset == 'breast_cancer':
        x, y_true, similarities, label_dict = load_hypbc_multi_group(
                        num_groups=num_groups,
                        num_data_samples = args.num_data_samples,
                        feature_dim = args.feature_dim,
                        method = args.similarity_metric,
                        feature_correlation_thresh = args.feature_correlation_thresh,
                        visualize = True)
    else:
        assert (False,"only breast cancer dataset possible")

    for i in range(len(x)):
        train_internal(args, x[i], y_true[i], similarities[i], label_dict,prefix=f"mul_{i}")


def single_train(args):
    # create dataset
    label_dict = None
    if args.dataset == 'breast_cancer':
        x, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=1,
                                                                     num_data_samples=args.num_data_samples,
                                                                     feature_dim=args.feature_dim,
                                                                     method=args.similarity_metric,
                                                                     feature_correlation_thresh=args.feature_correlation_thresh,
                                                                     visualize=True)
    else:
        x, y_true, similarities, label_dict = load_data(args.dataset, data_size=args.num_data_samples)

    train_internal(args,x[0], y_true[0], similarities[0], label_dict, prefix="sin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser = add_flags_from_config(parser, config_args)
    args = parser.parse_args()
    single_train(args)

