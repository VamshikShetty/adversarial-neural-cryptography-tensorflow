
import tensorflow as tf
import numpy as np

learning_rate   = 0.0008
batch_size      = 4096
sample_size     = 4096*5 # 4096 according to the paper
epochs          = 10000  # 850000 according to the paper
steps_per_epoch = int(sample_size/batch_size)

# BOB_LOSS_THRESH = 0.02  # Exit when Bob loss < 0.02 and Eve > 7.7 bits
# EVE_LOSS_THRESH = 7.7


# Input and output configuration.
TEXT_SIZE = 16
KEY_SIZE  = 16

# Training parameters.
ITERS_PER_ACTOR = 1
EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

# Set a random seed to help reproduce the output
seed = 7919
tf.set_random_seed(seed)
np.random.seed(seed)

# False if we want to train from scratch and true to contiune training a already trained model
restore_trained_model = False


def random_bools(sample_size, n):

  temp =  np.random.random_integers(0, high=1, size=[sample_size, n])
  temp = temp*2 - 1
  return temp.astype(np.float32)
  

def model(collection, message, key=None):

  if key is not None:
    combined_message = tf.concat(axis=1, 
      values=[message, key])
  else:
    combined_message = message

  with tf.variable_scope(collection):
    fc    = tf.layers.dense( combined_message, TEXT_SIZE + KEY_SIZE, activation=tf.nn.relu)
    fc = tf.expand_dims(fc, 2)

    # tf.contrib.layers.conv1d( input, filters, kernel_size, stride, padding, activation_fn)
    conv1 = tf.layers.conv1d( fc, filters= 2, kernel_size= 4, strides= 1, padding='SAME',  activation=tf.nn.sigmoid)
    conv2 = tf.layers.conv1d( conv1, filters= 4, kernel_size= 2, strides=2, padding='VALID', activation=tf.nn.sigmoid)
    conv3 = tf.layers.conv1d( conv2, filters= 4, kernel_size= 1, strides=1, padding='SAME',  activation=tf.nn.sigmoid)

    # output
    conv4 = tf.layers.conv1d( conv3, filters= 1, kernel_size= 1, strides=1, padding='SAME',  activation=tf.nn.tanh)


    out   = tf.squeeze(conv4, 2)
  return out



Alice_input_message  = tf.placeholder(tf.float32, shape=(batch_size, TEXT_SIZE), name='Alice_input_message')
Alice_input_key      = tf.placeholder(tf.float32, shape=(batch_size, KEY_SIZE), name='Alice_input_key')



Alice_out_cipher = model('Alice', Alice_input_message, Alice_input_key)
Bob_out_message  = model('Bob', Alice_out_cipher, Alice_input_key)
Eve_out_message  = model('Eve', Alice_out_cipher)


## Eves LOSS
Eves_loss = (1/batch_size)*tf.reduce_sum( tf.abs( Eve_out_message - Alice_input_message ))

## ALICE AND BOB LOSS
Bob_loss = (1/batch_size)*tf.reduce_sum( tf.abs( Bob_out_message  - Alice_input_message ))
Eve_evadropping_loss = tf.reduce_sum( tf.square(float(TEXT_SIZE) / 2.0 - Eves_loss) / ((TEXT_SIZE / 2)**2) )

Alice_bob_loss = Bob_loss + Eve_evadropping_loss



# Get tensors to train
Alice_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Alice') 
Bob_vars   =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Bob') 
Eve_vars   =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eve') 

Eve_opt  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=1e-08).minimize(Eves_loss, var_list=[Eve_vars])
bob_opt  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=1e-08).minimize(Alice_bob_loss, var_list=[Alice_vars + Bob_vars])


sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

alice_saver = tf.train.Saver(Alice_vars)
bob_saver   = tf.train.Saver(Bob_vars)
eve_saver   = tf.train.Saver(Eve_vars)


if restore_trained_model:
  alice_saver.restore(sess, "weights/alice_weights/model.ckpt")
  bob_saver.restore(sess, "weights/bob_weights/model.ckpt")
  eve_saver.restore(sess, "weights/eve_weights/model.ckpt")


# DATASET 
messages = random_bools(sample_size, TEXT_SIZE)
keys     = random_bools(sample_size, KEY_SIZE)


# Training begins
for i in range(epochs):

  for j in range(steps_per_epoch):

    # get batch dataset to train
    batch_messages = messages[j*batch_size: (j+1)*batch_size]
    batch_keys     = keys[j*batch_size: (j+1)*batch_size]

    # Train Alice and Bob
    for _ in range(ITERS_PER_ACTOR):
      temp = sess.run([bob_opt, Bob_loss, Eve_evadropping_loss, Bob_out_message],feed_dict={Alice_input_message:batch_messages , Alice_input_key:batch_keys })
      
      temp_alice_bob_loss = temp[1]
      temp_eve_evs_loss   = temp[2]
      temp_bob_msg        = temp[3]

    # train Eve
    for _ in range(ITERS_PER_ACTOR*EVE_MULTIPLIER):
      temp = sess.run([Eve_opt, Eves_loss, Eve_out_message], feed_dict={Alice_input_message:batch_messages , Alice_input_key:batch_keys })

      temp_eve_loss = temp[1]
      temp_eve_msg  = temp[2]

  # save after every 500 epochs
  if i%500 == 0 and i!=0:
    alice_saver.save(sess, "weights/alice_weights/model.ckpt")
    bob_saver.save(sess, "weights/bob_weights/model.ckpt")
    eve_saver.save(sess, "weights/eve_weights/model.ckpt")


  # output bit error and loss after every 100 epochs
  if i%50 == 0:
    print('  epochs: ', i, '  bob bit error: ', temp_alice_bob_loss,' + ', temp_eve_evs_loss,'   & eve bit error:', temp_eve_loss)

sess.close()
