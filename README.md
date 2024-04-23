# Image-to-Image Retrieval System Using FAISS
Image retrieval systems are becoming popular daily as the demand for image search increases. Since images can be represented as feature vectors, they can be easily stored in the vector database for faster query retrieval. This paper focuses on extracting image features from the SBU Captions dataset using a pre-trained SigLIP (Sigmoid Loss for Language Image Pre-Training) model and evaluating the FAISS (Facebook AI Similarity Search) library for retrieving k similar images for a given input image. The final results and performance of the image retrieval system are assessed using different metrics, including Recall at K and system latency. The system's latency is measured using brute force Euclidean distance and various FAISS indexes.

## Pre-Requisites:
1. Download the SBU Captions Dataset from http://www.cs.rice.edu/~vo9/sbucaptions/sbu_images.tar  
2. Install Python3 and required libraries as per the code file
3. Download the 1 Million FAISS IndexFlatL2 file from https://drive.google.com/file/d/1i77SSTKCtDipQaCvr08yCEWgLAJUjYPU/view?usp=share_link
4. Download the 1 Million FAISS IndexIVFFlat file from https://drive.google.com/file/d/19Ob35sH_iNT8Zqu66mwA44JREj2t1K-4/view?usp=share_link


## Code Structure:

**job.slurm** -> This is the job that we run on Rice University's NOTS server. It contains the configurations of node on which the ImageFeatureExtractor.py will get executed. We execute a job in `nots.rice.edu` server using `sbatch job.slurm submit` command after doing `ssh` into the instance. 

**ImageFeaturesExtractor.py** -> Use this file to extract the vectors of fixed dimensions from images using the desired SigLIP model and save it to csv file for later usage. Since there were 1 million images, so we splitted the extraction of image features into mulitple csv files, each containing 100K image vectors or rows in csv.  

**IndexCreation.ipynb** ->  For each extracted csv file, we can create the Faiss Index using this notebook. The advantage of creating index file is that it compresses the csv file drastically and can be loaded in the future easily (less time consuming) for reconstruction of image vectors.

**Merge100k.ipynb** -> Once the 100k Indexes are created using the IndexCreation.ipynb, we need to merge the indexes in a single index file. Using this code, we can merge the indexes together into a single Index consisting of 1 Million vectors.  

**LatencyMesaurement.ipynb** -> This file is used to measure the time taken to retrieve k similar images from FAISS index consisting of 1M index. First we read the merged index files and reconstruct the image vectors and then measure the time taken to create the index and search within the index. We measure the latency for FAISS's IndexFlatL2 and IndexIVFFlat.

**Recall.ipynb** -> This notebook is used to first generate a set of Random Images from the SBU captions dataset. This code takes both IndexFlatL2 and IndexIVFFlat FAISS indexes of 1 Million vectors as input and calculates Recall @ K metric for K = 1,10,20,30..., 100 for each of the indexes.
