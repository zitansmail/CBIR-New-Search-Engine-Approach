"""

@author: Zitane Smail
"""


from sklearn.decomposition import PCA
import numpy as np
import joblib
import os

from annoy import AnnoyIndex

from feautures import get_feauture_vector, load_pretrained_vit



#Applaying Annoy - and pca to reduce the verctor dementions
def build_annoy_index(db_path, cfg):
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
    
    feutures_shape =  cfg["modele"]["feutures_shape"]
    n_components =  cfg["PCA"]["n_components"]
    n_three =  cfg["Annoy"]["n_tree"]
    print("[INFO] Loading Features")
    feature_list = np.load(os.path.join(db_path, 'feautures.npy')).reshape(-1,feutures_shape)

    print("[INFO] Running PCA")
    # 128 default value
    n_components = n_components
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(feature_list)
    joblib.dump(pca, os.path.join(db_path, "pca.joblib"))

    print("[INFO] Building Index")
    feature_length = n_components
    index = AnnoyIndex(feature_length, 'angular')
    for i, j in enumerate(components):
        index.add_item(i, j)

    index.build(n_three)
    index.save(os.path.join(db_path, "index.annoy"))
    
    
#get Similar images path 
def get_similar_images_path(input_path, model ,new, output_path=r"C:/Users/SecondArticle/corel Database", features_path=r"C:/Users/SecondArticle/corel Database", index_path=r"C:/Users/SecondArticle/corel Database", n=20):
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
    
    print("[INFO] Instantiating Model")
    resultPath= []
    print("[INFO] Loading Image Filename Mapping")
    filename_list = np.load(os.path.join(features_path, "filesnames.npy"))

    print("[INFO] Extracting Feature Vector")
    model = load_pretrained_vit()
    image = get_feauture_vector(input_path,model)
    #input_features = np.array(image)
    input_features = image

    print("[INFO] Applying PCA")
    
    pca = joblib.load(os.path.join(features_path, "pca.joblib"))
    components = pca.transform(input_features)[0]

    print("[INFO] Loading ANN Index")
    ann_index = AnnoyIndex(components.shape[0], 'angular')
    ann_index.load(os.path.join(index_path, "index.annoy"))

    print("[INFO] Finding Similar Images")
    indices = ann_index.get_nns_by_vector(components, n, search_k=-1, include_distances=False)
    indices = np.array(indices)
    similar_image_paths = filename_list[indices]

    print("[INFO] Saving Similar Images to {0}".format(output_path))
    feature_list = np.load(os.path.join(features_path, "feauturesCorelFull.npy")).reshape(-1,1024)
    feature_list = feature_list[indices]
   
    for idx, path in enumerate(similar_image_paths):
        #print(path)
        #break
        resultPath.append(path.split('/')[-1])
    return resultPath