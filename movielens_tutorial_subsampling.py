# modify from https://github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial.py
"""Training a deep NN on MovieLens with differentially private Adam optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

from tensorflow_privacy.privacy.optimizers import dp_optimizer
from gdp_accountant import *

#### FLAGS
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, '
                        'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .01, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.55,
                      'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 5, 'Clipping norm')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_float('max_mu', 2, 'Maximum mu before termination')
flags.DEFINE_string('subsampling', 'Poisson', 'Poisson or Uniform subsampling')

FLAGS = flags.FLAGS

microbatches=10000
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

n_users=6040
n_movies=3706

def nn_model_fn(features, labels, mode):
#https://github.com/hexiangnan/neural_collaborative_filtering
#https://nipunbatra.github.io/blog/ml/2017/12/29/neural-collaborative-filtering.html
#https://github.com/tiangolo/tensorflow-models/blob/master/official/recommendation/neumf_model.py
    n_latent_factors_user = 10
    n_latent_factors_movie = 10
    n_latent_factors_mf = 5
    
    user_input = tf.reshape(features['user'], [-1,1])
    item_input = tf.reshape(features['movie'], [-1,1])

    mf_embedding_user = tf.keras.layers.Embedding(n_users,n_latent_factors_mf,input_length=1)
    mf_embedding_item = tf.keras.layers.Embedding(n_movies,n_latent_factors_mf,input_length=1)
    mlp_embedding_user = tf.keras.layers.Embedding(n_users,n_latent_factors_user,input_length=1)
    mlp_embedding_item = tf.keras.layers.Embedding(n_movies,n_latent_factors_movie,input_length=1)
    
    # GMF part
    # Flatten the embedding vector as latent features in GMF
    mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
    mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
    # Element-wise multiply
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

    # MLP part
    # Flatten the embedding vector as latent features in MLP
    mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
    # Concatenation of two latent features
    mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])
    
    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])
    
    logits = tf.keras.layers.Dense(5)(predict_vector)    

    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)
  
    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
  
      if FLAGS.dpsgd:
        # Use DP version of GradientDescentOptimizer. Other optimizers are
        # available in dp_optimizer. Most optimizers inheriting from
        # tf.train.Optimizer should be wrappable in differentially private
        # counterparts by calling dp_optimizer.optimizer_from_args().
        optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=microbatches,
            learning_rate=FLAGS.learning_rate)
        opt_loss = vector_loss
      else:
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        opt_loss = scalar_loss
  
      global_step = tf.compat.v1.train.get_global_step()
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
      return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'rmse':
              tf.compat.v1.metrics.root_mean_squared_error(
                  labels=tf.cast(labels, tf.float32),
                  predictions=tf.tensordot(a=tf.nn.softmax(logits,axis=1),b=tf.constant(np.array([0,1,2,3,4]),dtype=tf.float32),axes=1))
      }
      return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def load_adult():
    import pandas as pd
    # https://grouplens.org/datasets/movielens/1m/
    data = pd.read_csv('data/ratings.dat', sep='::', header=None,names=["userId", "movieId", "rating", "timestamp"])
    n_users=len(set(data['userId']))
    n_movies=len(set(data['movieId']))
    print('number of movie: ',n_movies)
    print('number of user: ',n_users)

    # give unique dense movie index to movieId
    from scipy.stats import rankdata
    data['movieIndex']=rankdata(data['movieId'], method='dense')
    # minus one to reduce the minimum value to 0, which is the start of col index

    print('number of ratings:',data.shape[0])
    print('percentage of sparsity:',(1-data.shape[0]/n_users/n_movies)*100,'%')

    from sklearn.model_selection import train_test_split
    train,test=train_test_split(data,test_size=0.2,random_state=100)
    
    return train.values-1, test.values-1, np.mean(train['rating'])


def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(3)

  # Load training and test data.
  train_data, test_data, mean = load_adult()

  # Instantiate the tf.Estimator.
  adult_classifier = tf.estimator.Estimator(model_fn=nn_model_fn,
                                            model_dir=FLAGS.model_dir)

  # Create tf.Estimator input functions for the training and test data.
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={'user': test_data[:,0], 'movie': test_data[:,4]},
      y=test_data[:,2],
      num_epochs=1,
      shuffle=False)

  # Training loop.
  steps_per_epoch = 800167 // 10000
  test_accuracy_list = []
  for epoch in range(1, FLAGS.epochs + 1):
    np.random.seed(epoch)
    for step in range(steps_per_epoch):
        tf.compat.v1.set_random_seed(0)
        whether=np.random.random_sample(800167)>(1-10000/800167)
        subsampling=[i for i in np.arange(800167) if whether[i]]
        global microbatches
        microbatches=len(subsampling)

        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
          x={'user': train_data[subsampling,0], 'movie': train_data[subsampling,4]},
          y=train_data[subsampling,2],
          batch_size=len(subsampling),
          num_epochs=1,
          shuffle=True)
        # Train the model for one step.
        adult_classifier.train(input_fn=train_input_fn, steps=1)

    # Evaluate the model and print results
    eval_results = adult_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['rmse']
    test_accuracy_list.append(test_accuracy)
    print('Test RMSE after %d epochs is: %.3f' % (epoch, test_accuracy))
    
    # Compute the privacy budget expended so far.
    if FLAGS.dpsgd:
        if FLAGS.subsampling=='Poisson':
            eps = compute_epsP(epoch,FLAGS.noise_multiplier,800167,10000,1e-6)
            mu = compute_muP(epoch,FLAGS.noise_multiplier,800167,10000)
        if FLAGS.subsampling=='Uniform':
            eps = compute_epsU(epoch,FLAGS.noise_multiplier,800167,10000,1e-6)
            mu = compute_muU(epoch,FLAGS.noise_multiplier,800167,10000)
      
        print('For delta=1e-5, the current MA epsilon is: %.2f' % 
              compute_epsilon(epoch,FLAGS.noise_multiplier,800167,10000,1e-6))
        print('For delta=1e-5, the current CLT epsilon is: %.2f' % eps)
        print('For delta=1e-5, the current mu is: %.2f' % mu)
      
        if mu>FLAGS.max_mu:
            break
    else:
      print('Trained with vanilla non-private SGD optimizer')
    
if __name__ == '__main__':
  app.run(main)
