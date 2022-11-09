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



def load_pretrained_vit()->Sequential:
    """Run command in sub process.
    Runs the command in a sub process with the variables from `env`
    added in the current environment variables.
    Parameters
    ----------
    command: List[str]
        The command and it's parameters
    env: Dict
        The additional environment variables
    Returns
    -------
    int
        The return code of the command
    """
    print("[INFO] Start loading pre-trained model")
    vit_model = vit.vit_l32(
        image_size = (int(IMAGE_SIZE), int(IMAGE_SIZE)),
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False)
    return vit_model
    

# Extract feautures from giving image
def get_feauture_vector(file,model):
    """Run command in sub process.
    Runs the command in a sub process with the variables from `env`
    added in the current environment variables.
    Parameters
    ----------
    command: List[str]
        The command and it's parameters
    env: Dict
        The additional environment variables
    Returns
    -------
    int
        The return code of the command
    """
    #load image from giving path
    img = load_img(file, target_size=(int(IMAGE_SIZE), int(IMAGE_SIZE)))
    # convert from 'PIL.Image.Image' to numpy array
    img = img_to_array(img)
    #reshape the giving image
    reshaped_img = img.reshape(1,int(IMAGE_SIZE), int(IMAGE_SIZE), 3)
    #predict fauture vector
    return model.predict(reshaped_img, verbose=0)



def extracte_features(images : list, dirtosave, cfg):
    """Run command in sub process.
    Runs the command in a sub process with the variables from `env`
    added in the current environment variables.
    Parameters
    ----------
    command: List[str]
        The command and it's parameters
    env: Dict
        The additional environment variables
    Returns
    -------
    int
        The return code of the command
    """
    
    global IMAGE_SIZE, reshaped_image
    IMAGE_SIZE = cfg["modele"]["image_size"]
    model = load_pretrained_vit()
    print(type(model))
    extracted_feautures = {}
    try:
        for image in tqdm(images):
            prediction = get_feauture_vector(image, model)
            extracted_feautures[image] = prediction
            
        print('\n Save feautures and filenames')
        filenames = np.array(list(extracted_feautures.keys()))
        features = np.array(list(extracted_feautures.values()))
        
        np.save(os.path.join(dirtosave, "filesnames.npy"), filenames)
        np.save(os.path.join(dirtosave, "feautures.npy"), features)
    except Exception as exception:
        raise Exception("An error occured!", exception)