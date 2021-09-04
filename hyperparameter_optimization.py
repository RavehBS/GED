import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import optuna
import joblib
from datetime import  datetime
import copy
from math import comb


import optim
from config import config_args
from datasets.hc_dataset import HCDataset
from datasets.loading import load_data, load_hypbc_multi_group
from datasets.loading import load_hypbc
from model.hyphc import HypHC
from utils.metrics import dasgupta_cost
from utils.training import add_flags_from_config, get_savedir
from visualize import visualize_tree


class Objective(object):
    def __init__(self, args):
        # Hold this implementation specific arguments as the fields of the class.
        self.orig_args = args

    def __call__(self, trial):
        optim_args = copy.deepcopy(self.orig_args)

        #set all config hyperparams
        #general
        optim_args.epochs = trial.suggest_int("epochs",1,60,step=3)
        batch_size_power = trial.suggest_int("batch_size_power",6,9)
        optim_args.batch_size = 2**batch_size_power  # 64-512
        optim_args.learning_rate = trial.suggest_float("learning_rate",1e-5,1e-2,log=True)# logscale 1e-5 - 1e0

        # model
        optim_args.temperature = trial.suggest_float("temperature",0.001,0.2) # 0.01 - 0.5
        optim_args.init_size = trial.suggest_float("init_size",0.01,0.1) # 0.01-0.1
        optim_args.anneal_every = trial.suggest_int("anneal_every",10,100)
        optim_args.anneal_factor = ("anneal_factor",0.7,1.0) # 0.1-1.0

        # dataset
        optim_args.similarity_metric = trial.suggest_categorical("similarity_metric",['cosine','euclidean','mahalanobis','cityblock'])
        optim_args.feature_dim = trial.suggest_int("feature_dim", 10,200)  # 10-200

        #Init algorithm
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # get saving directory
        # TODO: consider not saving.
        if optim_args.save:
            save_dir = get_savedir(optim_args) + f"_trial{trial.number}"
            logging.info("Save directory: " + save_dir)
            save_path = os.path.join(save_dir, "model_{}.pkl".format(optim_args.seed))
            if os.path.exists(save_dir):
                if os.path.exists(save_path):
                    logging.info("Model with the same configuration parameters already exists.")
                    logging.info("Exiting")
                    return
            else:
                os.makedirs(save_dir)
                with open(os.path.join(save_dir, "config.json"), 'w') as fp:
                    json.dump(optim_args.__dict__, fp)
            log_path = os.path.join(save_dir, "train_{}.log".format(optim_args.seed))
            hdlr = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)

        # set seed
        logging.info("Using seed {}.".format(optim_args.seed))
        np.random.seed(optim_args.seed)
        torch.manual_seed(optim_args.seed)
        torch.cuda.manual_seed(optim_args.seed)

        # set precision
        logging.info("Using {} precision.".format(optim_args.dtype))
        if optim_args.dtype == "double":
            torch.set_default_dtype(torch.float64)

        # create dataset
        if optim_args.dataset == 'breast_cancer': #TODO: check how to optimize loading all the data each time
            x_all, y_true_all, similarities_all, label_dict = load_hypbc_multi_group(num_groups=1,
                                                                         num_data_samples=args.num_data_samples,
                                                                         feature_dim=args.feature_dim,
                                                                         method=args.similarity_metric,
                                                                         feature_correlation_thresh=args.feature_correlation_thresh,
                                                                         visualize=False)
            x = x_all[0]
            y_true = y_true_all[0]
            similarities = similarities_all[0]
        else:
            assert(False)

        print(similarities.shape)
        print(similarities)

        actual_num_samples = comb(len(y_true), 2) if optim_args.num_samples < 2 else optim_args.num_samples
        dataset = HCDataset(x, y_true, similarities, num_samples=actual_num_samples)
        dataloader = data.DataLoader(dataset, batch_size=optim_args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Generate the model.
        model = HypHC(dataset.n_nodes, optim_args.rank, optim_args.temperature, optim_args.init_size, optim_args.max_scale)
        model.to("cuda")

        # create optimizer
        Optimizer = getattr(optim, optim_args.optimizer)
        optimizer = Optimizer(model.parameters(), optim_args.learning_rate)

        # train model
        best_cost = np.inf
        best_model = None
        counter = 0
        logging.info("Start training")
        for epoch in range(optim_args.epochs):
            model.train()
            total_loss = 0.0
            with tqdm(total=len(dataloader), unit='ex') as bar:
                for step, (triple_ids, triple_similarities) in enumerate(dataloader):
                    # for param in model.parameters():
                    # print(param.data)
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
            if (epoch + 1) % optim_args.eval_every == 0:
                model.eval()
                tree = model.decode_tree(fast_decoding=optim_args.fast_decoding)

                # save embedding and weights for this epoch
                model_path = os.path.join(save_dir, f"model_sd{optim_args.seed}_epch{epoch}.pkl")
                torch.save(model.state_dict(), model_path)
                img_path = os.path.join(save_dir, f"embedding_sd{optim_args.seed}_epch{epoch}.png")
                visualize_tree(model, tree, y_true, img_path,label_dict)

                cost = dasgupta_cost(tree, similarities)

                logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))
                if cost < best_cost:
                    counter = 0
                    best_cost = cost
                    best_model = model.state_dict()
                else:
                    counter += 1
                    if counter == optim_args.patience:
                        logging.info("Early stopping.")
                        break

                trial.report(cost, epoch)  # report the values to optuna .
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # anneal temperature
            if (epoch + 1) % optim_args.anneal_every == 0:
                model.anneal_temperature(optim_args.anneal_factor)
                logging.info("Annealing temperature to: {}".format(model.temperature))
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= optim_args.anneal_factor
                    lr = param_group['lr']
                logging.info("Annealing learning rate to: {}".format(lr))

        logging.info("Optimization finished.")
        if best_model is not None:
            # load best model
            model.load_state_dict(best_model)

        if optim_args.save:
            # save best embeddings
            logging.info("Saving best model at {}".format(save_path))
            torch.save(best_model, save_path)

        # evaluation
        model.eval()
        logging.info("Decoding embeddings.")
        tree = model.decode_tree(fast_decoding=optim_args.fast_decoding)
        cost = dasgupta_cost(tree, similarities)
        logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))

        if optim_args.save:
            logger.removeHandler(hdlr)

        return best_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser = add_flags_from_config(parser, config_args)
    args = parser.parse_args()

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.

    hyper_param_log_path = "embeddings/breast_cancer/hyper_param"
    os.makedirs(hyper_param_log_path,exist_ok=True)
    logfile = os.path.join(hyper_param_log_path,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_hyper.log")
    logger.addHandler(logging.FileHandler(logfile, mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="hypbc", direction="minimize", sampler=sampler)
    logger.info("Start optimization.")
    study.optimize(Objective(args), n_trials=100, timeout=None)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
    joblib.dump(study, "study.pkl")






