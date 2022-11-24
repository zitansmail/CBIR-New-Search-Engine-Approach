# -*- coding: utf-8 -*-
"""

@author: Zitane Smail
"""


from feautures import *
from index_search import build_annoy_index
import glob
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    '''
    

    Parameters
    ----------
    cfg : DictConfig
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    '''
    dataset_path = cfg["environement"]["dataset_path"]
    feutures_path = cfg["environement"]["feutures_path"]
    dataset = glob.glob(f"{dataset_path}/**/*.jpg", recursive=True)
    extracte_features(dataset, feutures_path, cfg)
    build_annoy_index(feutures_path, cfg)

if __name__ == "__main__":
    my_app()
