# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:59:05 2020

@author: mousekinga82

The Neutral Style Transfer Practice Using VGG-19

Reference:
    "Very Deep Convolutional Networks For Large-Scale Image Recognition"
    by Karen Simonyan & Andrew Zisserman
    
    Coursea Deep.ai course by Andrew Ng.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from NST_util import *

learning_rate = 2
iter_num = 200
print_rate = 25
save_pc_prefix = "T6"

#Load Content image and then pre-prosessing
#content_image = plt.imread("images/louvre.jpg")
content_image = resize_image("images/persian_cat.jpg", CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
content_image = reshape_and_normalize_image(content_image)
#Save content photo H, W to CONFIG (Enable for arbitary size the same as content image)
#CONFIG.set_content_imageHW(content_image.shape[1], content_image.shape[2])

#Load & resize the sytle image to be the same as content image
style_image = resize_image("images/VanGogh_2.jpg", CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
#Pre-prosessing style image
style_image = reshape_and_normalize_image(style_image)

#Generate input image with noise
generated_image = generate_noise_image(content_image)

#Load pre-trained VGG model
tf.reset_default_graph()
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

#Get the total cost..
J_content = compute_content_cost(model, content_image, 'conv4_2')
J_style = compute_style_cost(model, style_image)
J_total = total_cost(J_content, J_style, alpha=1000, beta=1)

#Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
#Define train step
train_step = optimizer.minimize(J_total)

#Strat training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))
    for i in range(iter_num):
        sess.run(train_step)
        if i%print_rate == 0:
            Jt, Jc, Js, generated_image = sess.run([J_total, J_content, J_style, model['input']])
            J_ts = J_style
            print("Iteraiton " + str(i) + ":")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            #save image
            save_image("output/" + save_pc_prefix + "_" + str(i) + ".png", generated_image)
#save final image
save_image('output/' + save_pc_prefix + "_final.jpg", generated_image)

if (4 > 1):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(generated_image))
        a_G1, a_G2 = sess.run([model['conv1_1'], model['conv2_1']])
        sess.run(model['input'].assign(style_image))
        a_S1, a_S2 = sess.run([model['conv1_1'], model['conv2_1']])
        m, n_H, n_W, n_C = a_G1.shape
        s_cost1 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(a_S1, a_G1)))/(4 * np.square(n_H * n_W * n_C).astype('float32'))
        m, n_H, n_W, n_C = a_G2.shape
        s_cost2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(a_S2, a_G2)))/(4 * np.square(n_H * n_W * n_C).astype('float32'))
        s_cost =  40 * (s_cost1 + s_cost2)
        sa1, sa2 = sess.run([s_cost1, s_cost2])
        J__ts = sess.run(J_ts)