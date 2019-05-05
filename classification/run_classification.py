from random import shuffle              
import tensorflow as tf
import numpy as np
import os
from layers.graph_layer import chebyshev_polynomial_layer
from layers.anisotropic_graph_layer import ga_graph_layer
from layers.cubic_anisotropic_graph_layer import ga_cubic_graph_layer
from helper.helper import get_matrix_of_images
from helper.helper import get_images_names_from_folder
from classification.config import FishEyeSetup
from classification.config import SphericalSetup
from classification.config import CubicSetup
from classification.config import ModSphericalSetup
from classification.config import EXPER_NAME
from classification.config import RESTORE_FILE_NAME
from classification.config import N_CLASSES
from classification.config import N_CLASSES
from classification.config import CONV_LAYERS
from classification.config import FC_LAYERS
from classification.config import LOAD_PATH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, dest="exp_name", default="cubic")

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.005, shape=[size]))

def create_poolinf(layer):
    return tf.nn.max_pool(value=layer,
               ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1],
               padding='SAME')
 
    
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer


def get_feature_vector(inputs, width, height, batch_size, conv_filter_shapes, fc_filter_shapes, graph_building_name, is_cubic=False):
    with tf.variable_scope('Classification', reuse=tf.AUTO_REUSE):
        layer_ind = 0
        layer_input = inputs
        

        for filter_shape in conv_filter_shapes:
            input_shape = layer_input.shape
            convol = []
                
            if is_cubic:
                result_gal = ga_cubic_graph_layer(         
                            layer_input,
                            int(width / (2**layer_ind)),
                            int(height / (2**layer_ind)),
                            polynomial_degree=2,
                            pooling = 'Conv_stride2',
                            number_filters=filter_shape,
                            name_of_graph='get_agj_matrix_for_cubemap',
                            name="GraphLayer" + str(layer_ind))

            else:
                result_gal = ga_graph_layer(
                    layer_input,
                    int(width / (2**layer_ind)),
                    int(height / (2**layer_ind)),
                    polynomial_degree=2,
                    pooling = 'Conv_stride2',
                    name_of_graph=graph_building_name,
                    number_filters=filter_shape,
                    name="ChebEncoder" + str(layer_ind))

            result_gal = tf.nn.relu(result_gal)
            layer_input = result_gal
            layer_ind += 1                                 
            
        layer_input = tf.reduce_mean(layer_input, reduction_indices=[1,2])
        layer_input = tf.reshape(layer_input, (batch_size, -1))
        for filter_shape in fc_filter_shapes:
            input_shape = layer_input.shape
            result_gal = tf.nn.relu(result_gal)
            layer_input = tf.layers.dense(inputs=layer_input,
                                          units=filter_shape)            
    return layer_input   

def run_classification_task(batch_size,
                            image_shape,
                            number_of_iterations,
                            train_images, 
                            train_labels,
                            val_images, 
                            val_labels,
                            test_images, 
                            test_labels,
                            conv_fs,
                            fc_fs,
                            gfn,
                            n_labels,
                            is_train=True,
                            load_file=None,
                            restore_file=None,
                            images_by_name=False,
                            is_cubic=False
                           ):
    shape_batch_w_h_colors = [batch_size, image_shape[0], image_shape[1], image_shape[2]]
    shape_labels = [batch_size, n_labels]

    # print(train_images.shape)
    # print(val_images.shape)
    # print(test_images.shape)

    batch = tf.placeholder(tf.float32, shape_batch_w_h_colors)
    labels = tf.placeholder(tf.float32, shape_labels)
    
    feature_representation = get_feature_vector(batch, image_shape[1], image_shape[0],
                                                     batch_size, conv_fs, fc_fs, gfn, is_cubic=is_cubic)

    # define the loss function

    y = tf.nn.softmax(feature_representation)
    # clipped_output =  tf.clip_by_value(y, 1e-37, 1e+37)
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(clipped_output), reduction_indices=[1]))

    cross_entropy = tf.losses.softmax_cross_entropy(labels, feature_representation)

    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step_op  = tf.assign(global_step_tensor, global_step_tensor + 1)
    
    # define training step and accuracy
    tf_lr = tf.train.exponential_decay(0.005, global_step_tensor, 3, 0.9, staircase=True)
    tf.summary.scalar('tf_lr', tf_lr)   

    train_step = tf.train.AdamOptimizer(learning_rate=tf_lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.image('batch', batch)   
    tf.summary.scalar('accuracy', accuracy)           
    tf.summary.scalar('cross_entropy', cross_entropy)    
    merged = tf.summary.merge_all()
    
    # create a saver
    saver = tf.train.Saver()

    # initialize the graph
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # train    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    with tf.device("/gpu:0"):
        val_ac_best = -1
        test_ac_best = -1
        train_writer = tf.summary.FileWriter(load_file + '/train', sess.graph)

        sess.run(init)
        if not restore_file == None:
            with tf.name_scope('train'):
                saver.restore(sess, os.path.join(restore_file, "model.ckpt"))
        if is_train == True:
                train_writer = tf.summary.FileWriter(load_file + '/train', sess.graph)
                test_writer = tf.summary.FileWriter(load_file + '/test', sess.graph)
                iteration = 0
                ii = 0
                for _ in range(number_of_iterations):
                    if (ii > 0):
                        tf.get_variable_scope().reuse_variables()

                    iteration += 1
                    ii += 1
                    if (batch_size * (iteration + 1) >= len(train_images)):
                        iteration = 0
                        sess.run([inc_global_step_op])
                        
                    if images_by_name:
                        train_data = get_matrix_of_images(
                            train_images[batch_size * iteration:
                                         batch_size * (iteration + 1)]) / 255.
                    else:
                        train_data = train_images[batch_size * iteration:
                                                  batch_size * (iteration + 1)]
                    label_data = []
                    
                    for l in train_labels[batch_size * iteration: batch_size * (iteration + 1)]:
                        l_data = np.zeros(n_labels)
                        l_data[int(l)] = 1.
                        label_data.append(l_data)
                    
                    with tf.name_scope('train'):
                        ts, train_accuracy, summary = sess.run([train_step, accuracy, merged],
                                                  feed_dict={
                                batch: train_data,
                                labels: label_data})

                        train_writer.add_summary(summary, ii)                        
                    if ii % 150 == 0:
                        ta = []
                        iteration_val = 0
                        for _ in range(int(len(val_images) / batch_size) - 1):
                            iteration_val += 1
                            if (batch_size * (iteration_val + 1) >= len(val_images)):
                                iteration_val = 0
                                
                            if images_by_name:
                                val_data = get_matrix_of_images(
                                    val_images[batch_size * iteration_val:
                                               batch_size * (iteration_val + 1)]) / 255.
                            else:
                                val_data = val_images[batch_size * iteration_val:
                                                      batch_size * (iteration_val + 1)].reshape(batch_size,
                                                                                                image_shape[0],
                                                                                                image_shape[1],
                                                                                                image_shape[2])
                            label_data = []

                            for l in val_labels[batch_size * iteration_val: batch_size * (iteration_val + 1)]:
                                l_data = np.zeros(n_labels)
                                l_data[int(l)] = 1.
                                label_data.append(l_data)

                            with tf.name_scope('train'):
                                val_accuracy = sess.run(accuracy,
                                                          feed_dict={
                                            batch: val_data,
                                            labels: label_data})
                                ta.append(val_accuracy)
                        val_ac = sum(ta) / len(ta)
                        print("step %d, mean val accuracy: %f" % (ii, val_ac))
                        if val_ac > val_ac_best:
                            val_ac_best = val_ac
                            ta = []
                            test_iteration = 0
                            save_path = saver.save(sess, os.path.join(load_file, "model.ckpt"))
                            print("Model saved in path: %s" % save_path)
                            for _ in range(int(len(test_images)/batch_size) - 1):
                                test_iteration += 1
                                if images_by_name:
                                    test_data = get_matrix_of_images(
                                        test_images[batch_size * test_iteration:
                                                    batch_size * (test_iteration + 1)]) / 255. 
                                else:
                                    test_data = test_images[batch_size * test_iteration:
                                                            batch_size * (test_iteration + 1)].reshape(batch_size,
                                                                                                       image_shape[0],
                                                                                                       image_shape[1],
                                                                                                       image_shape[2])
                                label_data = []
                    
                                for l in test_labels[batch_size * test_iteration: batch_size * (test_iteration + 1)]:
                                    l_data = np.zeros(n_labels)
                                    l_data[int(l)] = 1.
                                    label_data.append(l_data)
                    
                                test_accuracy, test_sum = sess.run([accuracy, merged],
                                          feed_dict={
                                                batch: test_data,
                                                labels: label_data})
                                ta.append(test_accuracy)
                                test_writer.add_summary(test_sum, ii)
                            test_ac_best = sum(ta) / len(ta)
                            fi = open(os.path.join(load_file, "log.txt"), 'a+')
                            fi.write("mean test accuracy: %f \n" % (test_ac_best))
                            fi.close()
                            print("step %d, mean test accuracy: %f" % (ii, test_ac_best))

        fi = open(os.path.join(load_file, "log.txt"), 'a+')
        fi.write("Best mean test accuracy: %f" % (test_ac_best))
        fi.close()
        print(test_ac_best)

            
def run_cube_map_projection(conv_layers, fc_layers, name):
    for conv_fs, fc_fs, gfn in zip(conv_layers, fc_layers, graph_building_names):
        str_conv = "_".join([str(i) for i in conv_fs])
        str_fc_fs = "_".join([str(i) for i in fc_fs])
        tf.reset_default_graph()
        name += str(gfn)
        filename_for_load = os.path.join(LOAD_PATH, name + str_conv + "fc_" + str_fc_fs)

        # Load-images-labels
        test_images = np.array(np.load(CubicSetup.test_images), dtype=np.float32)
        test_images = test_images.reshape((-1, CubicSetup.n_nodes_phi, CubicSetup.n_nodes_theta, 1))
        test_labels = np.array(np.load(CubicSetup.test_labels), dtype=int)

        all_train_images = np.array(np.load(CubicSetup.all_train_images), dtype=np.float32)
        all_train_images = all_train_images.reshape((-1, CubicSetup.n_nodes_phi, CubicSetup.n_nodes_theta, 1))
        all_train_labels = np.array(np.load(CubicSetup.all_train_labels), dtype=int)

        val_images = all_train_images[-6000:]
        val_labels = all_train_labels[-6000:]
        train_images = all_train_images[:-6000]
        train_labels = all_train_labels[:-6000]

        run_classification_task(
            20,
            (n_nodes_phi, n_nodes_theta, 1),
            210000,
            train_images, 
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
            conv_fs,
            fc_fs,
            gfn,
            n_labels=N_CLASSES,
            is_train=True,
            restore_file=RESTORE_FILE_NAME,
            load_file= filename_for_load,
            images_by_name=False,
            is_cubic=True)
        
def run_fish_eye(conv_layers, fc_layers, name):
    for conv_fs, fc_fs, gfn in zip(conv_layers, fc_layers, graph_building_names):
        str_conv = "_".join([str(i) for i in conv_fs])
        str_fc_fs = "_".join([str(i) for i in fc_fs])
        tf.reset_default_graph()
        name += str(gfn)
        filename_for_load = os.path.join(LOAD_PATH, name + str_conv + "fc_" + str_fc_fs)
        
        # Load-images-labels
        test_images = np.array(np.load(FishEyeSetup.test_images), dtype=np.float32)
        test_images = test_images.reshape((-1, FishEyeSetup.n_nodes_phi, FishEyeSetup.n_nodes_theta, 1))
        test_labels = np.array(np.load(FishEyeSetup.test_labels), dtype=int)

        all_train_images = np.array(np.load(FishEyeSetup.all_train_images), dtype=np.float32)
        all_train_images = all_train_images.reshape((-1, FishEyeSetup.n_nodes_phi, FishEyeSetup.n_nodes_theta, 1))
        all_train_labels = np.array(np.load(FishEyeSetup.all_train_labels), dtype=int)

        val_images = all_train_images[-6000:]
        val_labels = all_train_labels[-6000:]
        train_images = all_train_images[:-6000]
        train_labels = all_train_labels[:-6000]

        run_classification_task(
            20,
            (n_nodes_phi, n_nodes_theta, 1),
            210000,
            train_images, 
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
            conv_fs,
            fc_fs,
            gfn,
            n_labels=N_CLASSES,
            is_train=True,
            restore_file=None,
            load_file= filename_for_load,
            images_by_name=False)

def run_spherical_projection(conv_layers, fc_layers, name):
    for conv_fs, fc_fs, gfn in zip(conv_layers, fc_layers, graph_building_names):
        str_conv = "_".join([str(i) for i in conv_fs])
        str_fc_fs = "_".join([str(i) for i in fc_fs])
        tf.reset_default_graph()
        name += str(gfn)
        filename_for_load = os.path.join(LOAD_PATH, name + str_conv + "fc_" + str_fc_fs + "spherical")
        # Load-images-labels
        test_images = np.array(np.load(SphericalSetup.test_images), dtype=np.float32)
        test_images = test_images.reshape((-1, SphericalSetup.n_nodes_phi, SphericalSetup.n_nodes_theta, 1))
        test_labels = np.array(np.load(SphericalSetup.test_labels), dtype=int)

        all_train_images = np.array(np.load(SphericalSetup.all_train_images), dtype=np.float32)
        all_train_images = all_train_images.reshape((-1, SphericalSetup.n_nodes_phi, SphericalSetup.n_nodes_theta, 1))
        all_train_labels = np.array(np.load(SphericalSetup.all_train_labels), dtype=int)

        val_images = all_train_images[-6000:]
        val_labels = all_train_labels[-6000:]
        train_images = all_train_images[:-6000]
        train_labels = all_train_labels[:-6000]
        
        run_classification_task(
                20,
                (n_nodes_phi, n_nodes_theta, 1),
                210000,
                train_images, 
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
                conv_fs,
                fc_fs,
                gfn,
                n_labels=N_CLASSES,
                is_train=True,
                restore_file=None,
                load_file= filename_for_load,
                images_by_name=False)

        
        
def run_spherical_mod_projection(conv_layers, fc_layers, name):
    for conv_fs, fc_fs, gfn in zip(conv_layers, fc_layers, graph_building_names):
        str_conv = "_".join([str(i) for i in conv_fs])
        str_fc_fs = "_".join([str(i) for i in fc_fs])
        tf.reset_default_graph()
        name += str(gfn)
        filename_for_load = os.path.join(LOAD_PATH, name + str_conv + "fc_" + str_fc_fs)
        
        tf.reset_default_graph()
        filename_for_load = filename_for_load + str(int(ModSphericalSetup.regul * 100))
            
        # Load-images-labels
        test_images = np.array(np.load(ModSphericalSetup.test_images), dtype=np.float32)
        test_images = test_images.reshape((-1, ModSphericalSetup.n_nodes_phi, ModSphericalSetup.n_nodes_theta, 1))
        test_labels = np.array(np.load(ModSphericalSetup.test_labels), dtype=int)

        all_train_images = np.array(np.load(ModSphericalSetup.all_train_images), dtype=np.float32)
        all_train_images = all_train_images.reshape((-1, ModSphericalSetup.n_nodes_phi, ModSphericalSetup.n_nodes_theta, 1))
        all_train_labels = np.array(np.load(ModSphericalSetup.all_train_labels), dtype=int)

        val_images = all_train_images[-6000:]
        val_labels = all_train_labels[-6000:]
        train_images = all_train_images[:-6000]
        train_labels = all_train_labels[:-6000]

        run_classification_task(
                20,
                (n_nodes_phi, n_nodes_theta, 1),
                210000,
                train_images, 
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
                conv_fs,
                fc_fs,
                gfn,
                n_labels=N_CLASSES,
                is_train=True,
                restore_file=None,
                load_file= filename_for_load,
                images_by_name=False)       
            
if __name__ == "__main__":
    
    args = parser.parse_args()
    exp_name = args.exp_name 

    if exp_name.lower() == 'cubic':
        EXPER_NAME = CubicSetup.exp_id
    elif exp_name.lower() == 'fisheye':
        EXPER_NAME = FishEyeSetup.exp_id
    elif exp_name.lower() == 'spherical':
        EXPER_NAME = SphericalSetup.exp_id
    elif exp_name.lower() == 'modspherical':
        EXPER_NAME = ModSphericalSetup.exp_id
    else:
        raise ValueError("Unknown experiment type")

    print("Classification has started [{:}]".format(exp_name))

    if EXPER_NAME == CubicSetup.exp_id:
        n_nodes_phi = CubicSetup.n_nodes_phi
        n_nodes_theta = CubicSetup.n_nodes_theta
        graph_building_names = CubicSetup.graph_building_names
        run_cube_map_projection(CONV_LAYERS, FC_LAYERS, 'CUBIC_')

    if EXPER_NAME == FishEyeSetup.exp_id:
        n_nodes_phi = FishEyeSetup.n_nodes_phi
        n_nodes_theta = FishEyeSetup.n_nodes_theta
        graph_building_names = FishEyeSetup.graph_building_names
        run_fish_eye(CONV_LAYERS, FC_LAYERS, 'FISHEYE_')

    if EXPER_NAME == SphericalSetup.exp_id:
        n_nodes_phi = SphericalSetup.n_nodes_phi
        n_nodes_theta = SphericalSetup.n_nodes_theta
        graph_building_names = SphericalSetup.graph_building_names
        run_spherical_projection(CONV_LAYERS, FC_LAYERS, 'SPHERICAL_')

    if EXPER_NAME == ModSphericalSetup.exp_id:
        n_nodes_phi = ModSphericalSetup.n_nodes_phi
        n_nodes_theta = ModSphericalSetup.n_nodes_theta
        graph_building_names = ModSphericalSetup.graph_building_names
        run_spherical_mod_projection(CONV_LAYERS, FC_LAYERS, 'SPHERICAL_MOD_')