
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tensorflow as tf

from skimage import feature
from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy.ndimage.filters import gaussian_filter


# In[2]:


def edge_detect(img):
    return roberts(img)

def gauss_filter(img, sigma=4):
    return gaussian_filter(img, sigma=sigma)

def invert(img):
    return img - 255

max_width = 100
max_height = 100
channels = 3


# In[3]:


def load_prep_img(path,sig=2):
    img = io.imread(path)
    oimg = img
    if len(img.shape) == 3:
        img = img[:,:,2]
    else:
        oimg = tf.stack([oimg,oimg,oimg], axis=2)
    img = edge_detect(img)
    img = np.stack([img,img,img], axis=2)
    img = gauss_filter(img,sigma=sig)
    return img,oimg
    
    


# In[4]:


def next_batch(batch_size,sig=2):
    pos_images = []
    proc_images = []
    orig_images = []
    if sig == 'rand':
        sig = np.random.randint(0,4)
    
    for _,_,files in os.walk('./images/'):
        for f in files:
            pos_images.append('./images/'+f)
    num_images = len(pos_images)
    random.shuffle(pos_images)
    for i in range(batch_size):
        x,y = load_prep_img(pos_images[i])
        x = tf.stack(x)
        y = tf.stack(y)
        x = tf.image.resize_image_with_crop_or_pad(x, max_height, max_width).eval()
        y = tf.image.resize_image_with_crop_or_pad(y, max_height, max_width).eval()
        proc_images.append(x)
        orig_images.append(y)
    return proc_images,orig_images


# In[5]:


# sample data
img,oimg = load_prep_img('./images/50.jpg',sig=np.random.randint(0,4))
fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(oimg)
a.set_title('Before')
a=fig.add_subplot(1,2,2)
plt.imshow(img)
a.set_title('After')


# In[6]:


# Training Parameters
learning_rate = 0.001
num_steps = 1
batch_size = 5

display_step = 1000
examples_to_show = 10


# In[7]:


X = tf.placeholder("float", [None, max_height,max_width,channels])
Y = tf.placeholder("float", [None, max_height,max_width,channels])


# In[8]:


#Building decoder (need to add deconv)
def encoder(x, mode=False):
    # convalution
    # input shape [batch_size,100,100,3]
    # output shape [batch_size, 100,100 6]
    conv1 = tf.layers.conv2d(
      inputs=x,
      filters=6,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name='conv1')
    
    # input shape [batch_size, 100,100 6]
    # output shape [batch_size, 50,50, 6]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    
    # input shape [batch_size, 50,50, 6]
    # output shape [batch_size, 50,50, 12]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=6,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name='conv2')
    
    # input shape [batch_size, 50,50, 12]
    # output shape [batch_size, 25,25, 12]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    print pool2

    pool2_flat = tf.reshape(pool2, [-1,25*25*6])
    x = tf.contrib.layers.fully_connected(pool2_flat,max_width*max_height*channels)
    
    dropout = tf.layers.dropout(inputs=x, rate=0.4, training=mode)
    # Decoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
    #                               biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                               biases['decoder_b2']))
    return dropout


# In[9]:


# Building the encoder (need to add conv)
def decoder(x, mode=False):
    
    x = tf.contrib.layers.fully_connected(x,(max_width*max_height*channels)/2)
    x = tf.contrib.layers.fully_connected(x,max_width*max_height*channels)
    
    return x


# In[10]:


# Construct model
encoder_op = encoder(X, mode=True)
decoder_op = decoder(encoder_op, mode=True)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = tf.reshape(Y,[batch_size,max_width*max_width*channels])


# In[11]:


# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


# In[12]:


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[13]:


# Start Training
# Start a new TF session
avg_loss = []
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        x,y = next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: x,Y: y})
        # Display logs per step
        avg_loss.append(l)
        if len(avg_loss) > 100:
            avg_loss.pop(0)
        if i % 100 == 0:
            x,y = next_batch(1)
            out =  sess.run(decoder_op,feed_dict={X: x})
            out = tf.cast(tf.reshape(out,[50,50,3]),'float')
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            plt.imshow(np.array(out.eval()))
        al = np.mean(avg_loss)
        sys.stdout.write("Step: %i || Loss: %f  || Avg Loss: %f \r" % (i,l,al) )
        sys.stdout.flush()


# In[ ]:



    

