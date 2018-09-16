#!/Users/majian/anaconda/bin/python
import cv2
import tensorflow as tf
import VGG
import config
import numpy as np
import os

if __name__=='__main__':
    #build graph
    name_op = tf.placeholder(dtype = tf.string)
    image_contents = tf.read_file(name_op)
    image = tf.image.decode_jpeg(image_contents, channels = 3)
    image = tf.image.resize_images(image, (config.IMG_H, config.IMG_W))#need to focus
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image,(-1,config.IMG_H, config.IMG_W,3))#need to focus
    logits_op = VGG.VGG16(image, config.N_CLASSES, True, False)
    prob_op = tf.nn.softmax(logits_op)
    #input image
    name = os.path.join('./testimg','dog.jpg')
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./logs2/train/')
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess,ckpt.model_checkpoint_path)
            logits = sess.run(logitis_op, feed_dict={name_op:name})
            print('logits:\n', logits)
            prob = sess.run(prob_op ,feed_dict = {name_op:name})[0,:]
            label, prob = np.argmax(prob), np.max(prob)
            print(label,prob)
        else:
            print('no checkpoint found!!!')
