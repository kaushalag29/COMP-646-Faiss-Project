# Image-to-Image Retrieval System Using FAISS

## Code Structure:

**ImageFeaturesExtractor.py** -> Use this file to extract the vectors of fixed dimensions from images using the desired SigLIP model and save it to csv file for later usage. Since there were 1 million images, so we splitted the extraction of image features into mulitple csv files, each containing 100K image vectors or rows in csv.  

**IndexCreation.ipynb** ->  For each extracted csv file, we can create the Faiss Index using this notebook. The advantage of creating index file is that it compresses the csv file drastically and can be loaded in the future easily (less time consuming) for reconstruction of image vectors.  

**LatencyMesaurement.ipynb** -> This file is used to measure the time taken to retrieve k similar images from FAISS index consisting of 1M index. First we read the merged index files and reconstruct the image vectors and then measure the time taken to create the index and search within the index. We measure the latency for FAISS's IndexFlatL2 and IndexIVFFlat.
