from argparse import ArgumentParser
import deepchem as dc
import pandas as pd
import numpy as np
import torch
from pathlib import Path

def run_featurize(config):
    """Featurizes dataset once and saves it locally as .npz from .csv.

    Args:
        config (_type_): configuration with filepaths to datapaths
    """
    target = ["melting_point"]
    data: pd.DataFrame = pd.read_csv(config["data_path"])
    # Featurizer requires modification depending on model used.
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True) #use_edges=True, use_chirality=True
    data_dir: Path = Path(config["data_path"]).parent
    filename: str = Path(config["data_path"]).stem
    if "train" in filename:
        dataset_type: str = "train"
    elif "valid" in filename:
        dataset_type: str = "valid"
    elif "test" in filename:
        dataset_type: str = "test"
    else:
        raise NameError("Please give your filepath an appropriate name for the corresponding dataset.")
    data_path: Path = data_dir / "input_{}.npz".format(dataset_type)
    target_path: Path = data_dir / "target_{}.npz".format(dataset_type)
    feats: np.ndarray = featurizer.featurize(data["smiles"])
    targets: np.ndarray = data[target].to_numpy()
    track_idx = 0
    track_idxs: list = []
    for datapoint in feats:
        flag = np.size(datapoint)
        if not flag:
            track_idxs.append(track_idx)
        track_idx += 1
    # remove datapoint in featurized data and targets.
    feats: np.ndarray = np.delete(feats, track_idxs)
    targets: np.ndarray = np.delete(targets, track_idxs)
    # remove in .csv file as well!
    data: pd.DataFrame = data.drop(labels=track_idxs)
    data.to_csv(config["data_path"], index=False)

    print("Number of invalid datapoints: {}".format(len(track_idxs)))
    print("Number of datapoints: {}".format(len(feats)))

    np.savez_compressed(data_path, feats)
    np.savez_compressed(target_path, targets)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Filepath to data.")
    args = parser.parse_args()
    config = vars(args)
    run_featurize(config)