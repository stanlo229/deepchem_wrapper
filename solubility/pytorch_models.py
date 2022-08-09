from argparse import ArgumentParser
from tracemalloc import start
import deepchem as dc
from deepchem.metrics import recall_score, precision_score, auc, accuracy_score, f1_score, kappa_score
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time

def run_molgraphconv(config):
    """_summary_

    Args:
        config (_type_): _description_
    """
    # Target name from column header
    target = ["log10_x_sol"]
    # load numpy data
    input_train: np.ndarray = np.load(config["train_path"], allow_pickle=True)
    input_test: np.ndarray = np.load(config["test_path"], allow_pickle=True)
    # load target data
    target_train: np.ndarray = np.load(config["train_target_path"], allow_pickle=True)
    target_test: np.ndarray = np.load(config["test_target_path"], allow_pickle=True)

    model_dir: Path = Path(config["train_path"]).parent.parent / config["model"] / "log"

    input_train: np.ndarray = input_train["arr_0"]
    input_test: np.ndarray = input_test["arr_0"]
    target_train: np.ndarray = target_train["arr_0"]
    target_test: np.ndarray = target_test["arr_0"]

    # print(input_train[0].num_node_features)
    # print(input_train[0].num_edge_features)
    # model choice
    train_dataset = dc.data.NumpyDataset(input_train, target_train)
    if config["model"] == "GAT":
        model = dc.models.GATModel(n_tasks=1, mode="regression", batch_size=16, predictor_hidden_feats=256, tensorboard=True, model_dir=model_dir)
    elif config["model"] == "GCN":
        model = dc.models.GCNModel(n_tasks=1, mode="regression", batch_size=16, predictor_hidden_feats=256, tensorboard=True, model_dir=model_dir)
    elif config["model"] == "MPNN":
        model = dc.models.torch_models.MPNNModel(n_tasks=1, mode="regression", batch_size=16, predictor_hidden_feats=256, tensorboard=True, model_dir=model_dir)
    elif config["model"] == "AttentiveFP":
        model = dc.models.AttentiveFPModel(n_tasks=1, mode="regression", batch_size=16, predictor_hidden_feats=256, tensorboard=True, model_dir=model_dir)

    # training
    nb_epoch = 100
    start_train_time = time.time()
    print("START TRAINING: {}". format(start_train_time))
    # NOTE: default loss is L2 (RMSE) for "regression"
    # NOTE: default loss is SparseSoftmaxCrossEntropy for "classification"
    loss = model.fit(train_dataset, nb_epoch=nb_epoch)
    end_train_time = time.time()
    print("TIME REQUIRED TO TRAIN {} for {} epochs: {} seconds".format(config["model"], nb_epoch, end_train_time - start_train_time))
    
    # inference
    test_dataset = dc.data.NumpyDataset(input_test, target_test)
    test_predictions: np.ndarray = model.predict(test_dataset)

    # export predictions to stdout.txt
    predictions_test: pd.DataFrame = pd.DataFrame(test_predictions, columns=target)
    prediction_path: Path = Path(config["train_path"]).parent.parent / config["model"] / "test_stdout.txt"
    predictions_test.to_csv(prediction_path, index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, help="Filepath to training data.")
    parser.add_argument("--test_path", type=str, help="Filepath to test data.")
    parser.add_argument("--train_target_path", type=str, help="Filepath to training target data.")
    parser.add_argument("--test_target_path", type=str, help="Filepath to test target data.")
    parser.add_argument("--model", type=str, help="Choose from: GAT, GCN, MPNN, AttentiveFP")
    args = parser.parse_args()
    config = vars(args)
    run_molgraphconv(config)