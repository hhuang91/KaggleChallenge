# Kaggle Challenge - Ultrasound Nerve Segmentation

## Overview

This is a summary of my attemp at the [Ultrasound Nerve Segmentation](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/overview)

The tasks aim to perform bianary segmentation of Brachial Plexus (BP) in ultrasound image. An example is shown below

![example](./_image/example.png)

## Refined Aim

While the higher final score the better, given the limited time, it's probably going to be hard to beat top ranking scores. So I need a more achievable goal that can demonstrate I have applied deep learning in a meaningful way for this challenge.

Therefore, I submitted a **null test**, basically predicting zeros masks for all test data. If my approach was able to **have higher score than null test**, then the deep learning method I developed is able to extract useful information from the training data.

Here is the score to beat:

![Null Test](./_image/null_test.png)

## Data Inspection & Processing

Good understanding of data characteristics is key to successful deep learning methods. Therefore, I started by looking at the data provided. And one issue immediately jumpped out:

### Class imbalance issue

Upon first inspection of provided `train_mask.csv`, many of the masks are empty (**3312 out of 5635**). 

* This huge imbalance in negative class can cause network get stuck in local minima where it predicts **all zeros**.
* This is also evident in the null score which is already relatively high

Therefore, the first step is to divide data based on if BP is present inside an image, creating subsets of **"BP-present"** (2323 images) and **"BP-absent"** (3312).

Then the partition of datasets for training was based on subject ids (47 in total) instead of just index of images to maximally avoid cross contamination. The partitioned data are shown below

![Data Partition](./_image/data_partition.png)



### Poor data quality (contradictory labels)

I realized this issue after a couple failed attempts to train the network to perform well. Upone close inspection of some of the data provided, I realized that there are contradictory labels for some of the images:

* For some images from the same patient, although they are clearly showing similar structure, one can have a segmentation label while the another has empty label. An example is shown below

![Confounding label](./_image/confounding_image.png)

Upon search in the discussion section of the Kaggle challenge, other users have reported the [same issue](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/discussion/21081#123201). I re-implemented one of the suggested method for filtering out such data [[ref]([13_clean/0_filter_incoherent_images.ipynb](https://github.com/julienr/kaggle_uns/blob/master/13_clean/0_filter_incoherent_images.ipynb))], which is based on histogram matching with cosine distance for recoginizing similar images.

In the end, 2237 images were excluded from the data for training, validation, and test.

![Data Removal](./_image/data_removal.png)

### Pre-processing of data

Due to the time limit, preprocessing of data was focused on basics: normalization, resizing, and BCE weighting computation to facilitate training. Please refere to [Future Work](#Tasks for the Future) for what other preprocessing I would do if given more time.

All the following preprocessing steps are done on-the-fly when loading data, built into [customized data loader class](./data/dataHandler.py)

#### Normalization

A straight foward way of normalizing images would be to just divide by 255, because all images are in `uint8`. However, we can perform dataset-wise Z-score normalization after reading images to bring all data into relatively close and normalized distribution, which would make it easier for training.

$$x = \frac{x-\mu}{\sigma}$$

With mean and standard deviation accumateled from all images: $\mu = 99.4072, \sigma= 56.5949$

#### Resizing

The images and labels will be rescaled to size of nearest $(2^m,2^n)$ before passing into the network [$(580,420) \rarr (512,512)$]. There are two reasons for this:

* Typical network for segmentation, e.g. U-Net, involves series of downsampling and upsampling. Working with image size that's power of 2 insures the preservation of images size, and input scalability.
* Pytorch's interpolation function for such operation can introduce inherent errors and is [non-deterministic](https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842), which is worse when output size is not multiple or divisible by input size. Resizing image prior to inference can ensure interpolation within network operates with divisible sizes, reducing such error. 

#### BCE weighting Calculation

As mentioned before, the imbalance of positive pixels and negative pixels (even in non-empty masks) can cause network to take a "shortcut" of just predicting zeros. To balance this, a weighting term for BCE loss computation should be provided based on ratio between positive and negative pixels

* when loading the masks, count number of positive pixels and number of negative pixels, $N_{pos},N_{neg}$
* use $\frac{N_{neg}}{N_{pos}}$ as weight parameter for `torch.nn.functional.binary_cross_entropy_with_logits`

### Data Augmentation

The data provided, after removal of contradictory data, is realtively limited (3398). Therefore, to prevent overfitting, data augmentation strategies are implemented

The following strategies are also implemented to perform on-the-fly when loading the data, incorperated into [customized data loader class](./data/dataHandler.py)

* random flip: 50% chance of flipping data horizontally, followed by 50% chance of vertical flipping
* random rigid transformation: $\pm5$ pixel shift in $H,W$ dimension, $\pm 5$ degree of random rotation
* noise injection: Guassian noise with $\sigma=2$% of the mean of the image
* random crop: images are randomly cropped from [512,512] to [384,384] 
  * the choice of 384 is because it is $6\times 2^5$, where 5 is the depth of my current network maxpooling dimension reduction
  * Also, the cropping does not affect network performance because CNN is input-scalable. So applying trained network on 512x512 images would still work

Note that except noise injection, other augmentations are performed on both image and mask to preserve spatial alignment, i.e. the integrity of labels

## Approach

Due to the clear challenge of vast presence of empty masks, directly train a network would be difficult. Therefore, I tried the following two approaches. (I thought of a thrid approach but didn't have the time to try it out, please refer to [Future work section](#Tasks for the Future))

### 1. Two networks for two tasks:

* A binary classification network deciding if BP is inside the image
  * Trained on both BP-present and BP-absent images
* A binary segmentation network segment BP from BP-present images
  * Trained (mostly) BP-present images 

### 2. Mixing in empty masks:

* Every epoch the network would go through all BP-present data but only a specificed (small) number of BP-absent data that are randomly selected
* Once converged, do transfer learning with entire dataset

Either of these approaches would need to train a segmentation network that performs well on BP-present data, so I decided to start from there. 

Before we start training network, we also need to choose loss function

### Loss Functions

Binary Segmentation --> **wighted BCE loss** (to account for class imbalance)

* when loading the masks, count number of positive pixels and number of negative pixels, $N_{pos},N_{neg}$
* use $\frac{N_{neg}}{N_{pos}}$ as weight parameter for `torch.nn.functional.binary_cross_entropy_with_logits`

Also, **modified DICE** with squared denominator impelmentation for class imbalance [[ref](https://arxiv.org/pdf/1606.04797)] and also smoothing of 1 on both nominator and denominator
$$
DICE = \frac{2\sum pq +1}{\sum p^\mathbf 2 + \sum q^\mathbf 2 +1}
$$

DICE is more sentitive to small structures, while the BCE loss can be influenced more by the negative masks (although partially compensated by weighting). The combination of the two should yield more reliable segmentation results

## Segmentation Network

### Standard U-Net

As the "gold standard" approach in segmentation tasks, U-Net is often the first choice and a baseline for comparison. For quickly testing, I used a semi-customizable U-Net model from [Pytorch Hub](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/) as a starting point.

#### Overfitting problem

After a couple of attemps, the U-Net always overfits to the training, when training loss continuously going down while the validation loss goes up, as shown below. **Neither heavy data augmentation nor smaller network could help**

![Unet training](./_image/Unet_training.png)

This is likely due to a combination of **complex image features,** **relatively limited data**, **simple training label**. This caused network to **memorize the label** associated with each image instead of extracting generalizable features. 

### Solution: Customized Multi-Resolution Network with Dropout in-between Convolutional Layers

This is based on an architeture I previously used that worked well in registration problem. It has a multi-level, encoder-decoder design with inter-level communication. One big advantage here is that this network is configurable to have dropout layers in encoder, decoder, and bottleneck. 

![Customized Network](./_image/custom_network.png)

#### Resolved overfitting with dropouts

With dropouts added for all convolutional blocks (please refer to [training configuration](./trainingConfig.json) for detailed network configuration settings). The overfitting problem was resolved, and validation loss was consistently lower than training loss.

![Network Training](./_image/net_training.png)

The consistent lower validation loss is expected because dropout basically makes network to train sub networks at each inference during training, but for validation, once all dropouts were turned off, the network is an ensemble of sub networks and therefore, is more robust.

#### Test Results

The test dataset consists of 23 BP-present images, the customized network achived mean DICE score of **0.61** for all these cases.

Unfortunately, directly apply this network to BP-absent test data does not work: predictions consisted of false positive for all 78 BP-absent images. But this was to be expected, and would be addressed in the subsequent transfer learning stage

## Transfer Learning to Improve Negative Detection

Network was 



**Final Solution**

Divide data into subsets of "BP-present" and "BP-absent." Then, trian binary segmentation network in two steps :

1. Every epoch the network would go through all BP-present data but only a specificed (small) number of BP-absent data that are randomly selected [handled by customized customized data sampler function, defined in `./data/dataHandler.py`]
2. Network goes through entrie training dataset including empty ones to futher finetune the performance on true negatives

### Statisics

Input images: $\mu = 99.4071923455182, \sigma= 56.59492460345583$ (mean and accumulated from all images)

Labels: each label has different balance between poistive and negative class. For better BCE loss computation, the ratio of $\frac{N_{neg}}{N_{pos}}$ should be provided for each batch during training (to use in `BCEWithLogitsLoss`).

## Data Processing

### Partition of data

We partition datasets based on subject ids (47 in total) instead of just index of images to maximally avoid cross contamination. Recall that we also need to divide data based on if mask is empty or not.

(note that although the challenge provided test dataset, it does not include ground truth label, so it will be difficult to analyze the performance)

* Non-empty subset: 2323 images in total 
  * Training: **subject 1-41**, (2031 images)
  * Validation: **subject 42-47**, (259 images) 
  * Test: **subject 47** (33 images)
* Empty subset: 3312 images in total
  * Training: **subject 1-41** (2885 images)
  * Validation: **subject 46** (340 images)
  * Test: **subject 47** (87 images)

For validation and testing, we test with all $259+340 = 599$ and $87+33=120$ pairs respectively without separation.

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

Also, **modified DICE** with squared denominator impelmentation for class imbalance [[ref](https://arxiv.org/pdf/1606.04797)] and also smoothing of 1 on both nominator and denominator
$$
DICE = \frac{2\sum pq +1}{\sum p^\mathbf 2 + \sum q^\mathbf 2 +1}
$$

DICE is more sentitive to small structures, while the BCE loss can be influenced more by the negative masks (although partially compensated by weighting). The combination of the two should yield more reliable segmentation results

### Data augmentation

Three data augmentation strategies are available:

1. Random horitonal flip
   * Each image has a probability of $p$ being flipped horizontally
2. Random rigid trasnformation
   * Each image (and corresponding mask ) has a random rotation of $\pm\theta$ degrees
   * Each image  (and corresponding mask ) has random transalation of $\mathbf T$ pixels in both $H$ and $W$ dimension
3. Noise injection
   * Each image (image only, mask is left untouched) has injection of Guassian noise with $\sigma=s*\mu_{im}$ 

Noise injection ($s=0.02$) and random flip ($p = 0.5$) are used in training, because those produces images similar to realistic data. The ranndom rigid transformation involves additional interpolation and the padding of image may also cause image to be not realistic.

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