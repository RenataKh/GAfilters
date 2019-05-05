from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers.graph_layer import chebyshev_polynomial_layer
import math 
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.util.tf_export import tf_export


flags = tf.app.flags
FLAGS = flags.FLAGS


class GAGraphLayer(base.Layer):
    def __init__(self, width, height, n_channels, polynomial_degree, number_filters, phi_limits=(-math.pi/2, math.pi/2),
                 theta_limits=(0, 2 * math.pi), name=None, pooling=None,
                 name_of_graph='get_adj_matrix_for_directed_omni_ga', **kwargs):
        super(GAGraphLayer, self).__init__()

        self.pooling=pooling
        self.width = int(width)
        self.height = int(height)
        self.n_channels = n_channels        
        self.polynomial_degree = polynomial_degree
        self.number_filters = number_filters
        self.name_of_graph = name_of_graph
    
    def call(self, inputs):
        input_shape = inputs.shape
        fm_sets = []
        # Here could be done cycle with different graphs
        for y_min, y_max, x_min, x_max in self.limits:
            result = chebyshev_polynomial_layer(         
                        inputs,
                        self.width,
                        self.height,
                        polynomial_degree=self.polynomial_degree,
                        pooling = self.pooling,
                        number_filters=self.number_filters,
                        name_of_graph=self.name_of_graph,
                        diff_graphs_attribute = [y_min, y_max, x_min, x_max],
                        name="GAGraphLayer")
            fm_sets.append(result)
        result_stack = tf.stack(fm_sets, axis=0, name='result_stack')                    
        final_result = tf.tensordot(result_stack, self.ga_graph_combo_coefs, axes=[0,0])
        return final_result
        
            
    def build(self, input_shape):
        # Add coefficients here
        with tf.variable_scope("UIGraphLayer", reuse=tf.AUTO_REUSE):
            self.limits = [
                [-1.5, -0.5, -1.5, -0.5],
                [-1.5, -0.5, -0.5, 0.5],
                [-1.5, -0.5, 0.5, 1.5],
                [-0.5, 0.5, -1.5, -0.5],
                [-0.5, 0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5, 1.5],
                [0.5, 1.5, -1.5, -0.5],
                [0.5, 1.5, -0.5, 0.5],
                [0.5, 1.5, 0.5, 1.5]
            ]
            self.ga_graph_combo_coefs = tf.get_variable(name="ui_graph_combo_coefs",
                                                        shape=(len(self.limits)),
                                                        initializer=tf.random_uniform_initializer(-1, 1),
                                                        dtype=tf.float32, 
                                                        trainable=True)
        self.built = True


@tf_export('layers.fully_connected_layer_softmax')
def ga_graph_layer(inputs,
                   width,
                   height,
                   polynomial_degree,
                   number_filters,
                   name_of_graph="get_adj_matrix_for_directed_omni_ga",
                   name=None,
                   pooling=None,
                   reuse=tf.AUTO_REUSE):
    """
    Chebyshev Polynomial Layer
    """
    layer = GAGraphLayer(width, height, inputs.shape[3], polynomial_degree,
                         number_filters, pooling=pooling,
                         name=name, _scope=name, name_of_graph=name_of_graph)
    return layer.apply(inputs)
