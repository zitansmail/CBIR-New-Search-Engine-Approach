[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## Intro 
__This repository contains a CBIR (content-based image retrieval) system__

__Extract query image's feature, and retrieve similar ones from image database__

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='/images/approach.png'></img>


## Resume

In this system our approach is based on the ViT architecture for ex-
tracting features from the Corel images. Then, we used the PCA as a features selector
to minimize the dimensionality. Finally, we implemented the Annoy algorithm for
similarity searches.

### Feature Extraction

- Vision Architecture
  - [Full article](https://arxiv.org/pdf/2010.11929.pdf)


### Dimension Reduction
The curse of dimensionality told that vectors in high dimension will sometimes lose distance property
- [PCA](https://github.com/pochih/CBIR/blob/master/src/random_projection.py)



###  Evaluation

CBIR system retrieves images based on __feature similarity__

Category | Precision (%) | Recall (%) 
--- | --- 
Buses | 100 | 20 
Mountains | 100 | 20 
Beach | 90 | 18 
Elephants | 100 | 20 
Food | 100 | 20 
Flowers | 100 | 20 
Africa | 100 | 20 
Horses | 100 | 20 
Dinosaurs | 100 | 20 
Buildings | 100 | 20 
Average | 100 | 20 


### Author
Zitane Smail / [@zitansmail](http://zitansmail.com)