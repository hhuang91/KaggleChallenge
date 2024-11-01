# Kaggle Challenge - Ultrasound Nerve Segmentation

## Task overview

Perform bianary segmentation of Brachial Plexus in ultrasound image

* **Input**

  Ultrasound images 

* **Label**

  Bianary mask 

## To-Do List

-------------------------- Pre Training -----------------------

- [x] Data Inspection (statistics, normalization, preprocessing)
- [x] Data processing (partition, preprocessing pipeline)
- [x] Network design (architecture, activation funciton, output module)
- [ ] Training strategy (initalization, loss function, data augmentation, optimizer)

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

Additona packages: pandas, matplotlib, scipy, tqdm

## Data Inspection

### Class imbalance issue

Upon first inspection of provided `train_mask.csv`, many of the masks are empty (**3312 out of 5635**). This can cause problems if unprocessed (huge imbalance in class, and network can get relatively high score with predicting mostly **zero** values)

***Solution***: 

Divide data into subsets of "non-empty mask" and "empty mask." Then, trian network in two steps :

1. Every epoch the network would go through all non-empty labels but only a specificed (small) number of empty labels that are randomly selected [handled by customized customized data sampler function, defined in `./data/dataHandler.py`]
2. Network goes through entrie training dataset including empty ones to futher finetune the performance on true negatives

### Statisics

Input images: $\mu = 99.4071923455182, \sigma= 56.59492460345583$ (mean and accumulated from all images)

Labels: each label has different balance between poistive and negative class. For better BCE loss computation, the ratio of $\frac{N_{neg}}{N_{pos}}$ should be provided for each batch during training (to use in `BCEWithLogitsLoss`).

## Data Processing

### Partition of data

We partition datasets based on subject ids (47 in total) instead of just index of images to maximally avoid cross contamination. Recall that we also need to divide data based on if mask is empty or not.

(note that although the challenge provided test dataset, it does not include ground truth label, so it will be difficult to analyze the performance)

* Non-empty subset: 2323 images in total 
  * Training: **subject 1-41**, (2305 images)
  * Validation: **subject 42-47**, (328 images) 
  * Test: **subject 47** (33 images)
* Empty subset: 3312 images in total
  * Training: **subject 1-41** (2885 images)
  * Validation: **subject 46** (340 images)
  * Test: **subject 47** (87 images)

For testing, we test with all $87+33=120$ pairs without separation.

### Preprocessing

Not much preprocessing is needed, becuase the images all have same size, and thery are already in loadable format and realtive small (for future work, can nromalize and then convert them into `.pth` first for slightly faster loading time)

A straight foward way of normalizing images would be to just divide by 255, because all images are in `uint8`. However, we can perform Z-score normalization after reading images to bring all data into relatively close and normalized distribution, which would make it easier for training.
$$
x = \frac{x-\mu}{\sigma}
$$
In addition, the images and labels will be rescaled to size of nearest $(2^m,2^n)$ before passing into the network [$(580,420) \rarr (512,512)$]. The reason for this is that we use multi-resolution "U-Net-like" architecture that involves series of downsampling and upsampling. Pytorch's interpolation function for such operation can introduce inherent errors and is [non-deterministic](https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842), which is worse when output size is not multiple or divisible by input size. Resizing image prior to inference can ensure interpolation within network operates with divisible sizes, reducing such error. 

## Network Design

**Multi-resolution U-Net-Like structures**

## Training Strategy

### Initialization

Network is fully convolutional --> Using Kaiming (He) initialization

### Loss funciton

Binary Segmentation --> **wighted BCE loss** (to account for class imbalance)

* when loading the masks, count number of positive pixels and number of negative pixels, $N_{pos},N_{neg}$
* use $\frac{N_{neg}}{N_{pos}}$ as weight parameter for `torch.nn.functional.binary_cross_entropy_with_logits`

Also, **modified DICE** with squared denominator impelmentation for class imbalance [[ref](https://arxiv.org/pdf/1606.04797)]
$$
D = \frac{2\sum p }{\sum p^\mathbf 2 + \sum q^\mathbf 2}
$$

## Overfitting Test

## Training of Network

## Hyperparameter Tuning/Selection

## API

## Performance Evaluation

## Tasks for the Future

Dropout

axusilary task 

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