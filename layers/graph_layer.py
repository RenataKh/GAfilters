from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import layers.graph_builder as graph_builder
from layers.graph_builder import get_adj_matrix_for_omni_ga
from layers.graph_builder import get_adj_matrix_for_weighted_graph
from layers.graph_builder import get_adj_matrix_for_graph_regular_weightones
from layers.graph_builder import get_indexes_of_binary_drop_mask
from layers.graph_builder import get_binary_drop_mask
import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import tf_export
from scipy.linalg import circulant
from scipy import sparse
from random import randrange

import numpy as np
import math 

flags = tf.app.flags
FLAGS = flags.FLAGS


class ChebyshevPolynomialLayer(base.Layer):
    def __init__(self, width, height, n_channels, polynomial_degree, number_filters, phi_limits=(-math.pi/2, math.pi/2),
                 theta_limits=(-math.pi, math.pi), name=None, pooling=None,
                 name_of_graph='get_adj_matrix_for_omni_ga', diff_graphs_attribute=None, **kwargs):
        super(ChebyshevPolynomialLayer, self).__init__()
        
        self.pooling=pooling
        
        self.width = int(width)
        self.height = int(height)
        
        print ("w", self.width, "h", self.height)
        
        self.n_channels = n_channels        
        self.polynomial_degree = polynomial_degree
        self.number_filters = number_filters

        print (str(name_of_graph), diff_graphs_attribute)
        if diff_graphs_attribute == None:
            adj_matrix = getattr(graph_builder, name_of_graph)(self.height, self.width)
        elif name_of_graph == "get_agj_matrix_for_cubemap":
            adj_matrix = getattr(graph_builder, name_of_graph)(self.height,
                                                               self.width,
                                                               shift_hor=diff_graphs_attribute[0],
                                                               shift_ver=diff_graphs_attribute[1])

        else:
            adj_matrix = getattr(graph_builder, name_of_graph)(self.height,
                                                               self.width,
                                                               range_theta=theta_limits,
                                                               y_min=diff_graphs_attribute[0],
                                                               y_max=diff_graphs_attribute[1],
                                                               x_min=diff_graphs_attribute[2],
                                                               x_max=diff_graphs_attribute[3])
        
        if (self.pooling == 'Deconv_stride2'):
            adj_matrix = adj_matrix.transpose()        
        laplacian = self.get_laplacian(adj_matrix).tocoo()
        indices = np.mat([laplacian.row, laplacian.col]).transpose()
        self.tf_sparse_laplacian = tf.SparseTensor(indices, laplacian.data, laplacian.shape)

    def get_dot_product(self, matrix_1, matrix_2, sparse=False):
        """ Dot multiplication of two matrix: matrix_1 * matrix_2
            If matrix_1 is sparse put sparse=True
        """
        if sparse:
            return tf.sparse_tensor_dense_matmul(matrix_1, matrix_2)
        return tf.matmul(matrix_1, matrix_2)

    def get_distance(self, phi1, theta1, phi2, theta2):
        def get_x_y_z(phi, theta):
            z = np.sin(phi)
            y = np.cos(phi) * np.cos(theta)
            x = np.cos(phi) * np.sin(theta)
            return x, y, z
        x1, y1, z1 = get_x_y_z(phi1, theta1)
        x2, y2, z2 = get_x_y_z(phi2, theta2)
        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5
    
    def get_adjacency_matrix(self, width, height, phi_limits,
                             theta_limits, n_neighbours=[(-1, -1),(-1, 1),(1, -1),(1, 1),
                                                         (-1, 0),(1, 0),(0, -1),(0, 1)]):
        """ Build the weighted adjacency matrix graph for omnidirectional camera, where
            width, hight: is a size of images
            n_neighbours: is a neighbourhood to build a graph
        """
        def is_correct(ind_x, ind_y):
            if (ind_x < width and ind_y < height and ind_x >= 0 and ind_y >= 0):
                return True
            return False
        
        nodes_number = width * height
        adj_matrix = sparse.lil_matrix((nodes_number, nodes_number), dtype=np.float32)
        
        phi_step = (phi_limits[1] - phi_limits[0]) / height
        theta_step = (theta_limits[1] - theta_limits[0]) / width
        
        for y in range(height):
            phi = phi_limits[0] + phi_step * y
            for x in range(width):
                theta = theta_limits[1] + theta_step * x
                for delta_x, delta_y in n_neighbours:
                    if is_correct(x + delta_x, y + delta_y):
                        phi_neighbout = phi_limits[0] + phi_step * (y + delta_y)
                        theta_neighbout = theta_limits[0] + theta_step * (x + delta_x)
                        distance = self.get_distance(phi, theta, phi_neighbout, theta_neighbout)
                        adj_matrix[y * width + x, (y + delta_y) * width + (x + delta_x)] = 1. / (distance)
        return adj_matrix.tocoo()
    
    def get_laplacian_norm(self, adj_matrix):
        """ Return the Laplacian of the weigthed adjacency matrix
        """

        degree_matrix = adj_matrix.sum(axis=1, dtype=np.float32)    
        # Laplacian matrix:
        degree_matrix[degree_matrix == 0] = 1. 
        degree_matrix = 1.0 / np.sqrt(degree_matrix)
        degree_matrix = sparse.diags(np.ravel(degree_matrix), 0).tocsc()
        identity_matrix = sparse.identity(degree_matrix.size, dtype=np.float32)
        values = degree_matrix * adj_matrix * degree_matrix  
        values = values.multiply(1 / np.sum(values, axis=1))  # to avoid numerical problems
        return identity_matrix - values
    
    def get_laplacian(self, adj_matrix):
        """ Return the Laplacian of the weigthed adjacency matrix
        """

        degree_matrix = adj_matrix.sum(axis=1, dtype=np.float32)
        degree_matrix[degree_matrix == 0] = 1. 
        degree_matrix = sparse.diags(np.ravel(degree_matrix), 0).tocsc()
        # Laplacian matrix:
        L = degree_matrix - adj_matrix
        L = L.multiply(1.0 / np.sum(degree_matrix, axis=1))
        return L    
    
    def pixel_padding_for_deconvolutional(self, signal):
        # signal = batch x w x h x color_channels/fm
        shape_object = signal.shape

        # signal = batch color_channels/fm x h x w
        signal = tf.transpose(signal, (0,3,1,2))
        signal = tf.reshape(signal, (shape_object[0] * shape_object[3], shape_object[1], shape_object[2]))

        result = tf.concat([signal, tf.zeros_like(signal)], 2)
        result = tf.reshape(result, [shape_object[0] * shape_object[3], -1, 1])
        result = tf.concat([result, tf.zeros_like(result)], 2)
        result = tf.reshape(result, [shape_object[0], shape_object[3], shape_object[1]*2, -1])
        result = tf.transpose(result, (0,2,3,1))
        return result
        
    def get_chebyshev(self, tf_signal_vector):
        """ Return Chebyshev polynomial of degree polynomial_degree
            Input signals: tf_signal_vector -- batch_size x image_size (W x H) x "n_filter/color_channel"
            Laplacian matrx: self.tf_sparse_laplacian -- image_size (W x H) x image_size (W x H)            
        """
        if self.polynomial_degree == 1:
            with tf.variable_scope("ChebPolynomial", reuse=tf.AUTO_REUSE):
                return tf.multiply(tf_signal_vector, self.polynomial_coefs)
            
        # Change signal's shape to image_size (W x H) x batch_size*"n_filter/color_channel"
        signal_shape = tf_signal_vector.shape
        tf_signal_vector = tf.transpose(tf_signal_vector, (1,0,2))
        tf_signal_vector = tf.reshape(tf_signal_vector, (signal_shape[1], -1))
        
        # Apply chebyshev polybnomial filters        
        signal_1 = tf_signal_vector
        signal_2 = self.get_dot_product(self.tf_sparse_laplacian, signal_1, sparse=True)     
        # signal_2 = tf.transpose(tf_signal_vector, (1, 2, 0))
            
        stack = [signal_1, signal_2] 
        for k in range(2, self.polynomial_degree):
            signal_3 = 2 * self.get_dot_product(self.tf_sparse_laplacian,
                                                signal_2, sparse=True)  -  signal_1
            signal_1, signal_2 = signal_2, signal_3
            stack.append(signal_2)
        tf_chebyshhev_matrix = tf.stack(stack, axis=1)
        
        # Multiplying to the coefficients
        # tf_chebyshhev_matrix -- [(W x H) x polynomial_degree x batch_size*"n_filter/color_channel]
        with tf.variable_scope("ChebPolynomial", reuse=tf.AUTO_REUSE):
            tf_chebyshhev_matrix = tf.reshape(tf_chebyshhev_matrix, (signal_shape[1], self.polynomial_degree,
                                                                     signal_shape[0], signal_shape[2]))
            print (tf_chebyshhev_matrix.shape, self.polynomial_coefs.shape)
            tf_chebyshhev_matrix = tf.tensordot(tf_chebyshhev_matrix, self.polynomial_coefs, axes = [[1, 3], [0, 1]])
            # tf_chebyshhev_matrix -- image_size (H x W) x batch_size x number_filters
            tf_chebyshhev_matrix = tf.transpose(tf_chebyshhev_matrix, (1,0,2))
            return tf_chebyshhev_matrix        
        
    def apply_interpolation_mask(self, inputs):
        mask_a = np.zeros((self.height,self.width), dtype='float32')
        mask_a[::2,::2] = 1.0

        mask_b = np.zeros((self.height,self.width), dtype='float32')
        mask_b[::2,1::2] = 1.0

        mask_c = np.zeros((self.height,self.width), dtype='float32')
        mask_c[1::2,::2] = 1.0

        mask_d = np.zeros((self.height,self.width), dtype='float32')
        mask_d[1::2,1::2] = 1.0
        

        mask = np.asarray([mask_a.flatten(),mask_b.flatten(),mask_c.flatten(),mask_d.flatten()] * self.number_filters)
        mask = mask.transpose(1, 0)

        tensor_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        tensor_mask = tf.reshape(tensor_mask, (1, self.height * self.width, 4 * self.number_filters))
        
        result = tf.multiply(inputs, tensor_mask)
        result = tf.reshape(result, (-1, self.height * self.width, 4, self.number_filters))
        result = tf.reduce_sum(result, axis=2)        
        return result
    
    def call(self, inputs):
        input_init = inputs
        if (self.pooling == 'Deconv_stride2'):
            inputs = self.pixel_padding_for_deconvolutional(inputs)

        
        shapes = inputs.shape
        inputs = tf.reshape(inputs, (shapes[0], shapes[1] * shapes[2], shapes[3]))
        inputs = self.get_chebyshev(inputs)
        inputs = tf.reshape(inputs, (shapes[0], shapes[1], shapes[2], self.number_filters))
                    
        if (self.pooling == 'Conv_stride2'):
            inputs = inputs[:,::2,::2,:]
        return inputs
        
            
    def build(self, input_shape):
        with tf.variable_scope("ChebPolynomial", reuse=tf.AUTO_REUSE):
            print(tf.get_default_graph()._name_stack)

            self.polynomial_coefs = tf.get_variable(name="polynomial_coefs",
                                                      shape=(self.polynomial_degree, self.n_channels,
                                                             self.number_filters),
                                                      initializer=tf.random_uniform_initializer(-0.005, 0.005),
                                                      dtype=tf.float32, 
                                                      trainable=True)
        self.built = True    
        
    # def graph_spectral_conv(inputs,
    #                         polynomial_degree):
    #     layer = ChebyshevPolynomialLayer(width=inputs.shape[1],
    #                                      hight=inputs.shape[2],
    #                                      polynomial_degree=polynomial_degree)
    #     return layer.apply(inputs)

    
@tf_export('layers.fully_connected_layer_softmax')
def chebyshev_polynomial_layer(inputs,
                               width,
                               height,
                               polynomial_degree,
                               number_filters,
                               name_of_graph='get_adj_matrix_for_omni_ga',
                               name=None,
                               pooling=None,
                               diff_graphs_attribute=None,
                               reuse=tf.AUTO_REUSE):
    """
    Chebyshev Polynomial Layer
    """
    layer = ChebyshevPolynomialLayer(width, height, inputs.shape[3], polynomial_degree,
                                     number_filters, pooling=pooling,
                                     name=name, _scope=name, name_of_graph=name_of_graph,
                                    diff_graphs_attribute=diff_graphs_attribute)
    return layer.apply(inputs)
