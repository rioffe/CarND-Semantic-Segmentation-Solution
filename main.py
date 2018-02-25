import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.misc import toimage, imresize
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
from glob import glob
import re
from random import *
import scipy.misc

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
   
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

tests.test_load_vgg(load_vgg, tf)

beta = 0.003

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    output = tf.layers.conv2d_transpose(conv1x1, num_classes, 4, 2, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    
    conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    input = tf.add(output, conv1x1)
    input = tf.layers.conv2d_transpose(input, num_classes, 4, 2, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    
    
    conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    input = tf.add(input, conv1x1)
    input = tf.layers.conv2d_transpose(input, num_classes, 16, 8, padding='same', 
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
    
    return input

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    w1 = [v for v in tf.trainable_variables() if v.name == "conv2d/kernel:0"][0]
    w2 = [v for v in tf.trainable_variables() if v.name == "conv2d_transpose/kernel:0"][0]
    w3 = [v for v in tf.trainable_variables() if v.name == "conv2d_1/kernel:0"][0]
    w4 = [v for v in tf.trainable_variables() if v.name == "conv2d_transpose_1/kernel:0"][0]
    w5 = [v for v in tf.trainable_variables() if v.name == "conv2d_2/kernel:0"][0]
    w6 = [v for v in tf.trainable_variables() if v.name == "conv2d_transpose_2/kernel:0"][0]
 
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) + tf.nn.l2_loss(w6);
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss + beta * regularizers)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        print('Epoch ', epoch)
        for image, label in get_batches_fn(batch_size):
             _, loss_val = sess.run([train_op, cross_entropy_loss],
                           feed_dict={keep_prob: 0.5, correct_label: label, input_image: image, learning_rate: 0.001})
             print('loss = ', loss_val)
    
    return

tests.test_train_nn(train_nn)

def pipeline_helper(img, sess, logits, keep_prob, input_image):
    img_shape = img.shape
    image_shape = (img_shape[0] + 16, img_shape[1], img_shape[2])
    image = imresize(img, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: np.expand_dims(image, 0)})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = toimage(mask, mode="RGBA")
    street_im = toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)

def pipeline(sess, logits, keep_prob, input_image):
    g = lambda img: pipeline_helper(img, sess, logits, keep_prob, input_image)
    return g

# Given the input image,  randomly darken the image. Return darkened image.
# Image is darkened by reducing the V channel of an HSV image. Note that the range to darken was found empirically 
def random_dark_image(im):
    h_, s, v = cv2.split(im)

    darken = randint(0, 125)
    v = np.where(v > darken, v - darken, 0)

    im = cv2.merge((h_, s, v))
    return im

def my_gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                image = random_dark_image(image) 

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 30
    batch_size = 5

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    clip = VideoFileClip('driving.mp4')

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_fn = my_gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        print("vgg_input: ", vgg_input)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        print("nn_last_layer: ", nn_last_layer)
    
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        print("tf.trainable_variables: ", tf.trainable_variables())
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables. 
        saver = tf.train.Saver()

        sess.run(init)
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input,
             correct_label, vgg_keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)
        
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
        # Restore variables from disk.
        #saver.restore(sess, "/tmp/model.ckpt")
        #print("Model restored.")

        # Apply the trained model to a video
        new_clip = clip.fl_image(pipeline(sess, logits, vgg_keep_prob, vgg_input))
    
        # write to file
        new_clip.write_videofile('result.mp4')

if __name__ == '__main__':
    run()
