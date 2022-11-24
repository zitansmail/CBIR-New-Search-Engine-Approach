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
    cfg : DictConfig
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    global IMAGE_SIZE
    IMAGE_SIZE = cfg["modele"]["image_size"]
    image_path = cfg["modele"]["file"]
    if os.path.exists(image_path) == False:
        raise Exception("Image Not Found")
    #return image_path
    results = get_similar_images_path(image_path, IMAGE_SIZE)
    print(results)



    


evaluate()
    

    


