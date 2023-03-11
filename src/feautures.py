"""

@author: Zitane Smail
"""

# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from vit_keras import vit
import tensorflow_addons as tfa
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.models import Sequential
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf




def loadPretrainedVit(IMAGE_SIZE : int)->Sequential:
    
    '''

     Parameters
     ----------
     IMAGE_SIZE : int
    
     Returns
     -------
     Sequential

     '''
    
    print("[INFO] Start loading pre-trained model")
    
    vit_model = vit.vit_l32(
        image_size = (int(IMAGE_SIZE), int(IMAGE_SIZE)),
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False)
    
    return vit_model
    

# Extract feautures from giving image
def getFeautureVector(file : str, IMAGE_SIZE : int ,model: Sequential )->Sequential:
    '''
    

    Parameters
    ----------
    file : str.
    IMAGE_SIZE : int.
    model : Sequential.

    Returns
    -------
    Sequential.

    '''
    
 
    #load image from giving path
    img = load_img(file, target_size=(int(IMAGE_SIZE), int(IMAGE_SIZE)))
    # convert from 'PIL.Image.Image' to numpy array
    img = img_to_array(img)
    #reshape the giving image
    img_array = tf.expand_dims(img, 0)
    img_array = img_array /255
    
    #predict fauture vector
    return model.predict(img_array, verbose=0)



def exctacteFeatures(images : list, dirtosave : str, cfg : DictConfig)->None:
    '''
    

    Parameters
    ----------
    images : list.
    dirtosave : str.
    cfg : DictConfig.

    Raises
    ------
    Exception.

    Returns
    -------
    None.

    '''
   
    
    
    IMAGE_SIZE = cfg["modele"]["image_size"]
    model = loadPretrainedVit(IMAGE_SIZE)
    extracted_feautures = {}
    try:
        for image in tqdm(images):
            prediction = getFeautureVector(image,IMAGE_SIZE, model)
            extracted_feautures[image] = prediction
            
        print('[INFO] Save feautures and filenames')
        filenames = np.array(list(extracted_feautures.keys()))
        features = np.array(list(extracted_feautures.values()))
        
        np.save(os.path.join(dirtosave, "filesnames.npy"), filenames)
        np.save(os.path.join(dirtosave, "feautures.npy"), features)
    except Exception as exception:
        raise Exception("An error occured!", exception)