# -*- coding: utf-8 -*-
"""

@author: Zitane Smail
"""


from feautures import *
from index_search import buildAnnoyIndex
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import sys



@hydra.main(version_base=None, config_path="../conf", config_name="config")
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
    exctacteFeatures(dataset, feutures_path, cfg)
    buildAnnoyIndex(feutures_path, cfg)

if __name__ == "__main__":
    my_app()
