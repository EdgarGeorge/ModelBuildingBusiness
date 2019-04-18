from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#slim = tf.contrib.slim




def _depthwise_separable_conv(inputs,
                            num_pwc_filters,
                            width_multiplier,
                            sc,
                            downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=width_multiplier,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn

def mobilenet(inputs,
          num_classes=5,
          is_training=True,
          width_multiplier=1,
          scope='MobileNet'):
    """ MobileNet
    More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
    Returns:
    logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
                '''
                net = slim.stack(net,
                    _depthwise_separable_conv, 
                    [(64, width_multiplier, 'conv_ds/conv_ds_2'),
                    (128, width_multiplier, 'conv_ds/conv_ds_3', True),
                    (128, width_multiplier, 'conv_ds/conv_ds_4'),
                    (256, width_multiplier, 'conv_ds/conv_ds_5', True),
                    (256, width_multiplier, 'conv_ds/conv_ds_6'),
                    (512, width_multiplier, 'conv_ds/conv_ds_7', True),
                    (512, width_multiplier, 'conv_ds/conv_ds_8'),
                    (512, width_multiplier, 'conv_ds/conv_ds_9'),
                    (512, width_multiplier, 'conv_ds/conv_ds_10'),
                    (512, width_multiplier, 'conv_ds/conv_ds_11'),
                    (512, width_multiplier, 'conv_ds/conv_ds_12'),
                    (1024, width_multiplier, 'conv_ds/conv_ds_13', True),
                    (1024, width_multiplier, 'conv_ds/conv_ds_14')])
                '''
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
                
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    end_points['squeeze'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
    predictions = slim.softmax(logits, scope='Predictions')

    end_points['Logits'] = logits
    end_points['Predictions'] = predictions

    return logits, end_points





def mobilenet_v1_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
    normalizer_fn=slim.batch_norm):
  """Defines the default MobilenetV1 arg scope.

  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    normalizer_fn: Normalization function to apply after convolution.

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc





if __name__ == "__main__":
    npInput = np.random.randn(100,224,224,3).astype(np.float32)
    inputs = tf.Variable(npInput, name='input')
    logits,end_points = mobilenet(inputs)
#mobilenet.default_image_size = 224


