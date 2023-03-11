"""

@author: Zitane Smail
"""

from feautures import *
from index_search import *
import hydra
from omegaconf import DictConfig, OmegaConf
import os



    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate(cfg : DictConfig)->list:
    '''
    

    Parameters
    ----------
    cfg : DictConfig.

    Raises
    ------
    Exception.

    Returns
    -------
    list.

    '''
    global IMAGE_SIZE
    IMAGE_SIZE = cfg["modele"]["image_size"]
    image_path = cfg["modele"]["file"]
    if os.path.exists(image_path) == False:
        raise Exception("Image Not Found")
    #return image_path
    results = getSimilarImagesPath(image_path, IMAGE_SIZE)
    print(results)



    
evaluate()
    

    


