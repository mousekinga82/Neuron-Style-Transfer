# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:11:02 2020

@author: mousekinga82

"""

import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class CONFIG:
    IMAGE_HEIGHT = 395
    IMAGE_WIDTH = 700
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    VGG_MODEL_PATH = 'pretrained-model/imagenet-vgg-verydeep-19.mat'
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    #Define layer used for style
    STYLE_LAYERS = [
    ('conv1_1',0.1),
    ('conv2_1',0.3),
    ('conv3_1',0.3),
    ('conv4_1',0.3)]
    #('conv5_1',0.0)]
    
    @classmethod
    def set_content_imageHW(cls, H, W):
        cls.IMAGE_HEIGHT, cls.IMAGE_WIDTH = (H, W)
        
def pp():
    print(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)

def load_vgg_model(path):
    """
    Argument:
        path for the pre-trained vgg model
    Return:
        dict['layer name': computational graph]
    
    Load pretrained model of VGG-19.
    The configuration of the VGG model is shown below:
    (Note that the maxpool layer is changed to avgpool by suggestion of papaer, 
    and the last FC layers are discarded)
        # is 'name' (f, f, Nc_prev, Nc)
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    
    """
    #Only 'layers' is useful in the data format
    vgg = scipy.io.loadmat(path)['layers']
    
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias for a given layer
        """
        Wb = vgg[0][layer][0][0][2][0]
        #W -> (f,f,Nc_pre, Nc) ; b -> (Nc, 1)
        W, b = (Wb[0], Wb[1])
        #check name
        assert expected_layer_name == vgg[0][layer][0][0][0][0]
        return W, b
    
    def _conv2d(pre_layer, layer, layer_name):
        """
        Return the Conv2D layer using weights and bias from VGG
        """
        W, b = _weights(layer, layer_name)
        #Using tf.contant to fix the value (no trainig on weight & bias)
        W = tf.constant(W, name = layer_name + '_weights')
        # reshape b from (Nc, 1 ) to (Nc, ) 
        b = np.squeeze(b)
        b = tf.constant(b, name = layer_name + '_bias')
        return tf.nn.bias_add(tf.nn.conv2d(pre_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME'), b)
        
    def _relu(conv2d_layer):
        """
        Return the ReLu function of a Conv2D input
        """
        return tf.nn.relu(conv2d_layer)
    
    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Reuturn the Conv2D + ReLu layer using weights and bias form VGG
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))
    
    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #Construct the graph of the model
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

def get_model_conv_params(layer_name):
    """
    Return the conv2D parameters (Constant)
    
    Argument :  
        
    layer_name (ex:conv2_1)
    
    Return:   (weights, bias)
    """
    with tf.Session() as sess:
        weights, b = sess.run([layer_name+'_weights:0', layer_name+'_bias:0'])
    return weights, b

def compute_content_cost(model, content_image_pre, content_layer):
    """
    Computes the content cost from the chosen layer
    
    Arguments:
    model -- model used for NST
    content_image_pre -- content image after pre-processing
    content_layer -- name of the choesn layer (ex:conv2_1)
    
    Returns:
    J_content -- a scalar tensor representing the content cost.
    """
    with tf.Session() as sess:
        #assign (initialize the input variable)
        sess.run(model['input'].assign(content_image_pre))   
        out = model[content_layer]
        #a_C is now 'constant'
        a_C = sess.run(out)
        #a_G is stll an un-evaluated tensor  
        a_G = out
    #Retrieve dimensoins 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    #Reshape a_C & a_G
    a_C_unrolled = tf.reshape(a_C, shape = [m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [m, -1, n_C])
    #Compute the content cost
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))/(4 * n_H * n_W * n_C)  
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- tensor of shape (n_C, n_H*n_W)
    
    Return:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    n_C, _ = A.get_shape().as_list()
    GA = tf.matmul(A, tf.transpose(A, perm = [1, 0]))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), activation of a hidden layer of style image S.
    a_G -- tensor of dimension (1, n_H, n_W, n_C), activation of a hidden layer of generated image G.
    
    Returns:
    J_style_layer == tensor of a scalar, the cost of the style of a layer.
    """
    #Retrieve dimensions
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    #Reshpae image to (n_C, n_H*n_W)
    a_S = tf.reshape(tf.transpose(a_S[0], perm = [2, 0, 1]), shape = (n_C, -1))
    a_G = tf.reshape(tf.transpose(a_G[0], perm = [2, 0, 1]), shape = (n_C, -1))
    #Compute gram matrix
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    #Compute the cost
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/(4 * np.square(n_H * n_W * n_C, dtype = 'float32'))
    
    return J_style_layer

def compute_style_cost(model, style_image_pre, style_layers = CONFIG.STYLE_LAYERS):
    """
    Computes the overall style cost from serveral chosen layers
    
    Arguments:
    model -- model used for NST
    style_image_pre -- Style image after pre-processing
    style_layers -- A python list containing tuple: (layer name, coefficient of the layer)
    
    Returns:
    J_style -- a scalar tensor representing the total style cost.
    """
    J_style = 0
    with tf.Session() as sess:
        #assign (initialize the input variable)
        sess.run(model['input'].assign(style_image_pre))
        #Loop over layer chosen for style image
        for layer_name, coeff in style_layers:
            #Get the value of the style image at a layer
            out = model[layer_name]
            a_S = sess.run(out)
            #Save computation grah for gernerated image for style of a layer
            a_G = out
            J_style_layer = compute_layer_style_cost(a_S, a_G)
            J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost
    J_style -- total style cost of the selected layer
    alpha -- hyperparameter weighting the importance of the content image
    beta -- hyperparameter weighting the importance of the style image
    
    Returns:
    J -- total cost
    """
    return alpha * J_content + beta * J_style

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    #Reshape image to (m, n_H, n_W, n_C) for VGG input form
    image = np.reshape(image, ((1,) + image.shape))
    #Subtract the mean to match the input of VGG
    image = image - CONFIG.MEANS
    
    return image

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generate a noisy image by adding random noise to content image
    """
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
    #Set the input image to be a weighted average of the content and noise image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

def save_image(path, image):
    #Un-normalize the image 
    image = image + CONFIG.MEANS
    #clip & save image
    image = np.clip(image[0], 0, 255).astype('uint8')
    plt.imsave(path, image)
    
def resize_image(path, H, W):
    """
    Resize an image to an arbitrary size 
    """
    original_image = Image.open(path)
    fit_and_resized_image = ImageOps.fit(original_image, (W, H), Image.ANTIALIAS)
    return np.array(fit_and_resized_image)