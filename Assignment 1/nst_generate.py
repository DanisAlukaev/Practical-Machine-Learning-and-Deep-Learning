import warnings
warnings.filterwarnings('ignore')

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
from datetime import date, datetime

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m = "Tensors a_C and a_G should be of the same shape!" 
    assert a_C.get_shape().as_list() == a_G.get_shape().as_list(), m
    n_B, n_H, n_W, n_C = a_C.get_shape().as_list()
    
    shape = [n_B, n_C, n_H * n_W]
    _a_C, _a_G = tf.reshape(a_C, shape), tf.reshape(a_G, shape)
    J_content = tf.reduce_sum(tf.square(_a_C - _a_G)) / (4 * n_H * n_W * n_C)
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m = "Tensors a_S and a_G should be of the same shape!" 
    assert a_S.get_shape().as_list() == a_G.get_shape().as_list(), m
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    shape = [n_H * n_W, n_C]
    a_S = tf.transpose(tf.reshape(a_S, shape))
    a_G = tf.transpose(tf.reshape(a_G, shape))

    GS, GG = gram_matrix(a_S), gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * (n_C * n_H * n_W) ** 2)
    return J_style_layer

def compute_style_cost(sess, model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # TODO: ask Aidar
        a_S = tf.convert_to_tensor(a_S)
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J

def start_session():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    return sess

def load_image(img_path):
    image = scipy.misc.imread(img_path)
    if img_path[-4:] == ".png":
        image = image[:, :, :3] 
    image = reshape_and_normalize_image(image)
    return image

def configure_content_loss(sess, model, content_image):
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)

    # TODO: ask Aidar if I can do so
    a_C = tf.convert_to_tensor(a_C)

    a_G = out
    J_content = compute_content_cost(a_C, a_G)
    return J_content

def configure_style_loss(sess, model, style_image):
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)
    return J_style

def configure_total_loss(sess, model, content_image, style_image, a=10, b=40):
    J_content = configure_content_loss(sess, model, content_image)
    J_style = configure_style_loss(sess, model, style_image)
    J = total_cost(J_content, J_style, alpha=a, beta=b)
    return J_content, J_style, J

def model_nn(sess, model, input_image, content_image, style_image, generated_path, num_iterations=200):
    J_content, J_style, J = configure_total_loss(sess, model, content_image, style_image)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    sess.run(tf.global_variables_initializer())
    sess.run(model["input"].assign(input_image))

    dirname = datetime.now().strftime("%d-%m-%Y %H-%M-%S") + "/"
    os.mkdir("output/" + dirname)

    print("Start of the training...")
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i % 1 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + dirname + str(i) + ".png", generated_image)
    save_image(generated_path, generated_image)

    return generated_image

def main():
    m = "The command should contain paths to content, style and generated images"
    assert len(sys.argv) == 4, m
    content_path, style_path, generated_path = sys.argv[1:]
    content_image, style_image = load_image(content_path), load_image(style_path)
    generated_image = generate_noise_image(content_image)
    print("Content and style images were loaded successfully!")
    
    sess = start_session()
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    print("VGG-19 weights were loaded successfully")

    model_nn(sess, model, generated_image, content_image, style_image, generated_path)
    
if __name__ == "__main__":
    main()