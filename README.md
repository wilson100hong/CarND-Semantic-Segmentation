# Semantic Segmentation

---

[//]: # (Image References)

[image1]: ./images/fcn.png "FNC-8"
[image2]: ./images/segmented_video.png "Segmented Video"
[image3]: ./images/original_video.png "Original Video"
[image4]: ./images/runs_1.png "Run 1"
[image5]: ./images/runs_2.png "Run 2"
[image6]: ./images/runs_3.png "Run 3"
---

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


## Rubics

### Build the Neural Network

#### Does the project load the pretrained vgg model?
The function `load_vgg` passed the test.

#### Does the project learn the correct features from the images?
The function `layers` is implemented based on FNC-8 and passed the test:

![Screenshot][image1]

Additionally, 

1. L2 regularizer has been applied to all convolution and transpose convolution layers, with
   - weights_initializer_stddev = 0.01
   - weights_regularized_l2 = 1e-3

2. Pool layers `pool-3` and `pool-4` has been scaled:

    ```
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')
    ```

#### Does the project optimize the neural network?
The function `optimize` is implemented and passed the test. 
Adam is used for optimizer and L2 regulaization is also consider in cost:
```
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 1
cross_entropy_loss = cross_entropy_loss_raw + reg_constant * sum(reg_losses)
```

#### Does the project train the neural network?
The function `train_nn` is implemented and passed the test. After `50` epochs the loss is:
```
Epoch: 50: ['0.248600', '0.225263', '0.235733', '0.224057', '0.254750', '0.210725', '0.233797', '0.239854', '0.257384', '0.217595', '0.239896', '0.231317', '0.235352', '0.233919', '0.238826', '0.233647', '0.219818', '0.230765', '0.224539', '0.233072', '0.229059', '0.236796', '0.229197', '0.255547', '0.231950', '0.240968', '0.227417', '0.220171', '0.236826', '0.216200', '0.219603', '0.234617', '0.227941', '0.216510', '0.236924', '0.223891', '0.230761', '0.222113', '0.257428', '0.239652', '0.222175', '0.245257', '0.222560', '0.224805', '0.218568', '0.233658', '0.249912', '0.237365', '0.236286', '0.246143', '0.235422', '0.230948', '0.223160', '0.222349', '0.240384', '0.231010', '0.223560', '0.221441']
```
For the loss printed while training, see `train.log`.


### Neural Network Training

#### Does the project train the model correctly?
The loss decrease over time (epochs).

#### Does the project use reasonable hyperparameters?
- epochs = 50
- batch size = 5

#### Does the project correctly label the road?
From eyeing the prediction looks good. Generally it can cover 80% of roads and no more 20% of non-road areas. See sampled images from `runs` directory.

![Screenshot][image4]

![Screenshot][image5]

![Screenshot][image6]


### (Optional) Apply to video
[Video] (https://youtu.be/qjusJ1yFhTU)

![Screenshot][image2]


[Original Video (from KITTI benchmark)](https://youtu.be/yvjddIwPISk)

![Screenshot][image3]