# Neuron-Style-Transfer
Neuron style transfer practice

## What you can do
  - By using the pre-trained model of vgg19, we can extract the features in the middle layers. Chose one layer to computer the content cost to determine the "position" while getting the style cost of serveral layers to obtain the "style" to get the final generated image.
  - Here is some examples :
  


## Before starting

  - Please downlaod VGG-19 pre-trained model from "https://www.vlfeat.org/matconvnet/pretrained/"
then put it in the path 'pretrained-model/imagenet-vgg-verydeep-19.mat'
  - Prepare content & style image you want to do the style transfer on in '/images' and change the corresponded variable in 'NST.py' for "content_image" and "style_image"
  - The output picture size will be the same as content image and stored in '/outputs'
  - Some default setting can be found in 'NST_util.py'
  - Run NST.py to start the code.

## Theory Review

## Reference:

    (1) "Very Deep Convolutional Networks For Large-Scale Image Recognition 
          by Karen Simonyan & Andrew Zisserman
    (2) Coursea Deep.ai course by Andrew Ng.
