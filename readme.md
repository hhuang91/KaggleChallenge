# Kaggle Challenge

## Task overview

...description of the task

* **Input**
* **Output**

## To-Do List

-------------------------- Pre Training -----------------------

- [ ] Data Inspection (statistics, normalization, preprocessing)
- [ ] Data processing (partition, preprocessing pipeline)
- [ ] Network design (architecture, activation funciton, output module)
- [ ] Training strategy (initalization, loss function, optimizer)

---------------------------- Training ---------------------------

- [ ] overfitting test
- [ ] train the network
- [ ] (optional) hyperparameter tuning (learning rate & batch size)

------------------------- Post Training -----------------------

- [ ] API script (for easy testing of performance on new data)
- [ ] Performance evaluation pipeline (determine the performance of network)
- [ ] Results summary & analysis (in Jupyter Notebook for better readability)

## Environment/Packages

Python: 3.9

NumPy: 1.23.5

Pytorch: 2.5 ('+cuda118')

Cuda: 11.8

Additona packages: pandas, matplotlib,  scipy, tqdm

## Data Inspection

## Data Processing

## Network Design

## Training Strategy

## Overfitting Test

## Training of Network

## Hyperparameter Tuning/Selection

## API

## Performance Evaluation

## Tasks for the Future

### Additional Loss Functions

### Additional Features Extracted from Input Data

e.g.

* Image gradient
* Edge
* Transformations of image (Fourier Transform, Gabor Filter, Wavelet Decomposition)
* Ultize other existing tools (e.g. [total segmentator](https://github.com/wasserth/TotalSegmentator))

### Unsupervised Pre-training via Autoencoder

* Jigsaw puzzle
* Filling blanks
* Image reconstruction

### More Hyperparameter Tuning

* Network architecture 
  * number of levels
  * number of channels in encoder/decoder
* Activation functions
* Normalization layers
* Optimizer
* (Random seed)

--> Can use [**NeverGrad**](https://facebookresearch.github.io/nevergrad/) for hyperparameter search

### Ablation Study

* loss function
* archietecture design

### Options for Per-Instance Optimization

Allows the output of the network to be further optimized after inference for better accuracy at the expense of computational time if needed

* optimization pipeline
* optimization objective function(s)
* seamless integration 

### Implementing Fully Sharded Data Parrelle

Parrallel training fully utilizing multiple GPUs (distribute both model and data into GPUs)

* Allows training of larger models
* Faster training with large batch size and parrallel computing