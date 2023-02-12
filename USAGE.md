## How to run the code?

Let me show you how to use the code.

It can divided to two parts.

### Part1: make your image database
When you clone the repository, it will look like this:

    ├── conf/               # Configuration file
    ├── database/           # Directory of saved files (features, filenames, annoy index, pca)
    ├── dataset/            # Directory of all your images
    ├── images/             # Results
    ├── outputs/            # Results
    ├── resukts/            # Directory where the results will be stored
    ├── src/                # Source files
    ├── README.md           # How to use the code
    └── USAGE.md            # Intro to the repo

you need to put your images into a training_set directory in __dataset/__

    ├── dataset/            
       ├── training_set/
            ├── class 1/ 
            ├── class 2/

            ├── class n/


__all your image should put into dataset/__

In this directory, each image class should have its own directory

see the picture for details:

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='/images/datasetMod.PNG' padding='5px'></img>

In my database, there are 10 classes, each class has its own directory,

and the images belong to this class should put into this directory.

### Part2: run the code
the starting file is called main.py which will handle features extraction, then applying the pca and finally build the annoy index which will be used after in order to retrieve similar images and save them under the results directory.


```python
python3 src/main.py
```


## Author
Zitane Smail / [@zitansmail](http://zitansmail.github.io/)