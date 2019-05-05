import numpy as np
import os

CLASS_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = os.environ.get('DATASET_PATH', os.path.join(CLASS_PATH, '..', 'data'))

class CubicSetup:
    exp_id = 0
    n_nodes_phi = 64
    n_nodes_theta = 96
    graph_building_names = ['get_agj_matrix_for_cubemap']

    # train / val / test image paths 
    test_images = os.path.join(DATASET_PATH, "MNISTcubic", "test_mnist_images.npy")
    test_labels = os.path.join(DATASET_PATH, "MNISTcubic", "test_mnist_labels.npy")
    all_train_images = os.path.join(DATASET_PATH, "MNISTcubic", "train_mnist_images.npy")
    all_train_labels = os.path.join(DATASET_PATH, "MNISTcubic", "train_mnist_labels.npy")

class FishEyeSetup:
    exp_id = 1
    n_nodes_phi = 64
    n_nodes_theta = 128
    graph_building_names = ['get_adj_matrix_for_directed_fisheye']
   
    # Load train / val / test images
    test_images = os.path.join(DATASET_PATH, "fisheye", "test_mnist_images64.npy")
    test_labels = os.path.join(DATASET_PATH, "fisheye","test_mnist_labels.npy")
    all_train_images = os.path.join(DATASET_PATH, "fisheye", "train_mnist_images64.npy")
    all_train_labels = os.path.join(DATASET_PATH, "fisheye","train_mnist_labels.npy")
    
class SphericalSetup:
    exp_id = 2
    n_nodes_phi = 64
    n_nodes_theta = 128
    graph_building_names = ['get_adj_matrix_for_directed_rand_projection_ga']
    
    # Load train / val / test images
    test_images = os.path.join(DATASET_PATH, "MNISTomni", "omniphi_m05pi05pi_theta_mpipi_test_mnist_images.npy")
    all_train_images = os.path.join(DATASET_PATH, "MNISTomni","omniphi_m05pi05pi_theta_mpipi_train_mnist_images.npy")
    test_labels = os.path.join(DATASET_PATH, "MNISTomni", "test_mnist_labels.npy")
    all_train_labels = os.path.join(DATASET_PATH, "MNISTomni", "train_mnist_labels.npy")

class ModSphericalSetup:
    exp_id = 3
    n_nodes_phi = 64
    n_nodes_theta = 128
    graph_building_names = ['get_adj_matrix_for_directed_rand_projection_ga']
    regul = 0.15
    
    # Load train / val / test images
    test_images = os.path.join(DATASET_PATH, "MNISTrandom_projection","phi" + str(int(regul * 100)) + "test_mnist_images.npy")
    test_labels = os.path.join(DATASET_PATH, "MNISTrandom_projection","test_mnist_labels.npy")
    all_train_images = os.path.join(DATASET_PATH, "MNISTrandom_projection","phi" + str(int(regul * 100)) + "train_mnist_images.npy")
    all_train_labels = os.path.join(DATASET_PATH, "MNISTrandom_projection","train_mnist_labels.npy")

    
RESTORE_FILE_NAME = None    
N_CLASSES = 10
CONV_LAYERS = [[16, 32, 128]]
FC_LAYERS = [[512,  N_CLASSES]]
EXPER_NAME = CubicSetup.exp_id
LOAD_PATH = os.path.join(CLASS_PATH, '..', 'results')

if not os.path.exists(LOAD_PATH):
    os.makedirs(LOAD_PATH)