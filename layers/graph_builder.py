import numpy as np
import math 
from scipy import sparse
from scipy.linalg import circulant
import random
import numpy as np

# face_width = 32
# face_height = 32

# W = face_width
# H = face_height

# REGUL = 0.15


def get_phi_theta(x, y, radius, atan_of_radius, phi_i, theta_i):
    phi = math.asin(math.cos(atan_of_radius) * math.sin(phi_i) + y / radius * math.sin(atan_of_radius) * math.cos(phi_i))
    theta = theta_i + math.atan(x * math.sin(atan_of_radius) / (radius * math.cos(phi_i) * math.cos(atan_of_radius) 
                                                                 + y * math.sin(phi_i) * math.sin(atan_of_radius) + 1e-10))
    
    for shift in [0., 1., -1.]:
        xn, yn = phitheta2xy(phi, theta + shift * math.pi, phi_i, theta_i)
        if (((xn - x)**2 < 1e-10) and ((yn - y)**2 < 1e-10)):
            theta = theta + shift * math.pi
            break
    
    return phi, theta
    
def xy2phitheta(ind_x, ind_y, phi_i, theta_i):
    ro = (ind_x**2. + ind_y**2.) ** 0.5 + 1e-10
    c = math.atan(ro)
    return get_phi_theta(ind_x, ind_y, ro, c, phi_i, theta_i)

def phitheta2xy(phi_i, theta_i, phi_0, theta_0):
    cs = math.sin(phi_i) * math.sin(phi_0) + math.cos(phi_i) * math.cos(phi_0) * math.cos(theta_i - theta_0)
    if cs == 0:
        cs += 1e-10
    
    x = math.cos(phi_i) * math.sin(theta_i - theta_0) / cs
    y = (math.cos(phi_0) * math.sin(phi_i) - math.sin(phi_0) * math.cos(phi_i) * math.cos(theta_i - theta_0)) /cs
    
    return x, y

def phitheta2xy_np_stereographic(phi_i, theta_i, phi_0, theta_0, R=1.):
    """Fish-eye like"""
    k = 2 * R / (1. + np.sin(phi_0)*np.sin(phi_i) + np.cos(phi_i)*np.cos(phi_0)*np.cos(theta_i - theta_0) + 1e-10 )
    x = k * np.cos(phi_i) * np.sin(theta_i - theta_0)
    y = k * (np.cos(phi_0)*np.sin(phi_i) - np.sin(phi_0)*np.cos(phi_i)*np.cos(theta_i - theta_0))
    
    return x, y    

def phitheta2xy_np_stereographic(phi_i, theta_i, phi_0, theta_0, R=1.):
    """Fish-eye like"""
    k = 2 * R / (1. + np.sin(phi_0)*np.sin(phi_i) + np.cos(phi_i)*np.cos(phi_0)*np.cos(theta_i - theta_0))
    x = k * np.cos(phi_i) * np.sin(theta_i - theta_0)
    y = k * (np.cos(phi_0)*np.sin(phi_i) - np.sin(phi_0)*np.cos(phi_i)*np.cos(theta_i - theta_0))
    return x, y  

def get_radius_old(step_phi, step_theta):
    x, y = phitheta2xy(step_phi, step_theta, 0, 0)
    return (x**2 + y**2)**0.5

def get_distance(tp_phi, tp_theta, phi_n, theta_n):
    def get_point(phi, theta):
        x = math.cos(phi) * math.cos(theta)
        y = math.cos(phi) * math.sin(theta)
        z = math.sin(phi)
        return (x,y,z)
    x1,y1,z1=get_point(tp_phi, tp_theta)
    x2,y2,z2=get_point(phi_n, theta_n)
    return max(min(((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5, 1000), 0.0001)
                            

def get_index(phi_index, theta_index, n_nodes_theta):
    return phi_index * n_nodes_theta + theta_index


def get_indexes_of_binary_drop_mask(max_x, max_y):
    wsx = 7
    wsy = 7
    x = max_x / 2
    y = max_y / 2
    
    indexes = []
    for ind_x in range(wsx):
        ind_x = ind_x - wsx/2
        for ind_y in range(wsy):
            ind_y = ind_y - wsy/2
            if ind_x ==0 and ind_y==0:
                continue
            result_index = get_index(y + ind_y, x + ind_x, max_x)# - center_index
            indexes.append(result_index)
    return indexes

def get_binary_drop_mask(indexes, max_x, max_y, percent_taking_ind=0.25):
    num_to_select = int(len(indexes) * percent_taking_ind)
    string = ['0'] * max_x * max_y
    list_of_random_items = random.sample(indexes, num_to_select)
    for i in list_of_random_items:
        string[i] = '1'

    string = "".join(string)
    
    x = max_x / 2
    y = max_y / 2
    
    center_index = get_index(y, x, max_x)
    ind_string = string[center_index:] + string[:center_index]
    ind_for_matrix = [int(i) for i in ind_string]        
    return circulant(ind_for_matrix)


def get_adj_matrix_for_fisheye(n_nodes_phi, n_nodes_theta, phi_window=10, theta_window=15,
                              range_phi=(-math.pi / 2, math.pi / 2),
                              range_theta=(- math.pi, math.pi)):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    x, y = phitheta2xy_np_stereographic(tp_phi + delta_phi * step_phi,
                                       tp_theta + delta_theta * step_theta, tp_phi, tp_theta)

                    if (radius**2 >= x**2 + y**2):
                        # connect (tp_phi, tp_theta) with (tp_phi - delta_phi * step_phi, tp_theta - delta_theta * step_theta)   
                        if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                            theta_index_n = theta_index + delta_theta
                            phi_index_n = phi_index + delta_phi

                            if (theta_index_n < 0):
                                theta_index_n += n_nodes_theta

                            # process negative
                            if (phi_index_n >= 0) and (theta_index_n >= 0):
                                index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                                phi_n = phi_index_n * step_phi - math.pi / 2
                                theta_n = theta_index_n * step_theta

                                distance = get_distance(tp_phi, tp_theta, phi_n, theta_n)
                                adj_matrix[tp_index,index_neighbour] = 1. / distance
                                adj_matrix[index_neighbour,tp_index] = 1. / distance
                            
    return adj_matrix.tocoo()


def get_adj_matrix_for_omni_ga(n_nodes_phi, n_nodes_theta, phi_window=10, theta_window=15,
                              range_phi=(-math.pi / 2, math.pi / 2),
                              range_theta=(- math.pi, math.pi)):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    x, y = phitheta2xy(tp_phi + delta_phi * step_phi,
                                       tp_theta + delta_theta * step_theta, tp_phi, tp_theta)

                    if (radius**2 >= x**2 + y**2):
                        # connect (tp_phi, tp_theta) with (tp_phi - delta_phi * step_phi, tp_theta - delta_theta * step_theta)   
                        if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                            theta_index_n = theta_index + delta_theta
                            phi_index_n = phi_index + delta_phi

                            if (theta_index_n < 0):
                                theta_index_n += n_nodes_theta



                            # process negative
                            if (phi_index_n >= 0) and (theta_index_n >= 0):
                                index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                                phi_n = phi_index_n * step_phi - math.pi / 2
                                theta_n = theta_index_n * step_theta

                                distance = get_distance(tp_phi, tp_theta, phi_n, theta_n)
                                adj_matrix[tp_index,index_neighbour] = 1. / distance
                                adj_matrix[index_neighbour,tp_index] = 1. / distance
                            
    return adj_matrix.tocoo()


def get_adj_matrix_for_weighted_graph(n_nodes_phi, n_nodes_theta, phi_window=1, theta_window=1,
                                      range_phi=(-math.pi / 2, math.pi / 2),
                                      range_theta=(- math.pi, math.pi)):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                        theta_index_n = theta_index + delta_theta
                        phi_index_n = phi_index + delta_phi

                        if (theta_index_n < 0):
                            theta_index_n += n_nodes_theta

                        # process negative
                        if (phi_index_n >= 0) and (theta_index_n >= 0):
                            index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                            phi_n = phi_index_n * step_phi - math.pi / 2
                            theta_n = theta_index_n * step_theta

                            distance = get_distance(tp_phi, tp_theta, phi_n, theta_n)
                            adj_matrix[tp_index,index_neighbour] = 1. / distance
                            adj_matrix[index_neighbour,tp_index] = 1. / distance
                            
    return adj_matrix.tocoo()

def get_adj_matrix_for_graph_regular_weightones(n_nodes_phi, n_nodes_theta, phi_window=1, theta_window=1,
                                      range_phi=(-math.pi / 2, math.pi / 2),
                                      range_theta=(- math.pi, math.pi)):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)


            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                        theta_index_n = theta_index + delta_theta
                        phi_index_n = phi_index + delta_phi

                        if (theta_index_n < 0):
                            theta_index_n += n_nodes_theta

                        # process negative
                        if (phi_index_n >= 0) and (theta_index_n >= 0):
                            index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                            phi_n = phi_index_n * step_phi - math.pi / 2
                            theta_n = theta_index_n * step_theta

                            adj_matrix[tp_index,index_neighbour] = 1.
                            if (adj_matrix[index_neighbour,tp_index] == 0):
                                adj_matrix[index_neighbour,tp_index] = 1. 
                            
    return adj_matrix.tocoo()


def get_adj_matrix_for_directed_fisheye(n_nodes_phi, n_nodes_theta, phi_window=10, theta_window=15,
                              range_phi=(-math.pi / 2, math.pi / 2),
                              range_theta=(- math.pi, math.pi),
                              y_min=-1, y_max=1, x_min=-1, x_max=1):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
    y_min *= radius
    y_max *= radius
    x_min *= radius
    x_max *= radius
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    x, y = phitheta2xy_np_stereographic(tp_phi + delta_phi * step_phi,
                                       tp_theta + delta_theta * step_theta, tp_phi, tp_theta)
                    x *= -1
                    if y_min <= y < y_max and x_min <= x < x_max:
                            
                        if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                            theta_index_n = theta_index + delta_theta
                            phi_index_n = phi_index + delta_phi
                            if (theta_index_n < 0):
                                theta_index_n += n_nodes_theta

                            # process negative
                            if (phi_index_n >= 0) and (theta_index_n >= 0):
                                index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                                phi_n = phi_index_n * step_phi - math.pi / 2
                                theta_n = theta_index_n * step_theta

                                distance = get_distance(tp_phi, tp_theta, phi_n, theta_n)
                                adj_matrix[tp_index,index_neighbour] = 1. / distance
                            
    return adj_matrix.tocoo()



def get_adj_matrix_for_directed_omni_ga(n_nodes_phi, n_nodes_theta, phi_window=10, theta_window=15,
                              range_phi=(-math.pi / 2, math.pi / 2),
                              range_theta=(- math.pi, math.pi),
                              y_min=-1, y_max=1, x_min=-1, x_max=1):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius_old(step_phi,step_theta)
    y_min *= radius
    y_max *= radius
    x_min *= radius
    x_max *= radius
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    x, y = phitheta2xy(tp_phi + delta_phi * step_phi,
                                       tp_theta + delta_theta * step_theta, tp_phi, tp_theta)
                    x *= -1
                    if y_min <= y < y_max and x_min <= x < x_max:
                            
                        if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                            theta_index_n = theta_index + delta_theta
                            phi_index_n = phi_index + delta_phi
                            if (theta_index_n < 0):
                                theta_index_n += n_nodes_theta

                            # process negative
                            if (phi_index_n >= 0) and (theta_index_n >= 0):
                                index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                                phi_n = phi_index_n * step_phi - math.pi / 2
                                theta_n = theta_index_n * step_theta

                                distance = get_distance(tp_phi, tp_theta, phi_n, theta_n)
                                adj_matrix[tp_index,index_neighbour] = 1. / distance
                            
    return adj_matrix.tocoo()

def phitheta2xy_np_mod2(phi_i, theta_i, phi_0, theta_0, regul=0.1):
    
    add_theta_0 = regul*np.arcsin(np.sin(10*theta_0))
    add_theta_i = regul*np.arcsin(np.sin(10*theta_i))
    
    delta_add_theta = add_theta_i - add_theta_0
    
    cs = np.sin(phi_i) * np.sin(phi_0) + np.cos(phi_i) * np.cos(phi_0) * np.cos(theta_i - theta_0)
    
    x = np.cos(phi_i) * np.sin(theta_i - theta_0 + delta_add_theta) / cs
    y = (np.cos(phi_0) * np.sin(phi_i) - \
         np.sin(phi_0) * np.cos(phi_i) * np.cos(theta_i - theta_0)) / (1e-10 + cs)
    
    return x, y

def get_radius(step_phi, step_theta, projection):
    x, y = projection(step_phi, step_theta, 0,0)
    return (x**2 + y**2)**0.5

def phitheta2xy_np_mod(phi_i, theta_i, phi_0, theta_0, regul=0.2):
    
    add_phi_0 = regul*np.arcsin(np.sin(10*phi_0))
    add_phi_i = regul*np.arcsin(np.sin(10*phi_i))
    
    cs = np.sin(phi_i + add_phi_i) * np.sin(phi_0 + add_phi_0) + np.cos(phi_i) * np.cos(phi_0) * np.cos(theta_i - theta_0)
    
    x = np.cos(phi_i) * np.sin(theta_i - theta_0) / cs
    y = (np.cos(phi_0) * np.sin(phi_i + add_phi_i) - \
         np.sin(phi_0 + add_phi_0) * np.cos(phi_i) * np.cos(theta_i - theta_0)) / (1e-10 + cs)
    
    return x, y

def get_distance_mod(tp_phi, tp_theta, phi_n, theta_n, regul=0.2):
    def get_point(phi, theta):
        
        add_phi = regul*np.arcsin(np.sin(10*phi))
        
        x = math.sin(phi + add_phi) * math.cos(theta)
        y = math.sin(phi + add_phi) * math.sin(theta)
        z = math.cos(phi)
        return (x,y,z)
    x1,y1,z1=get_point(tp_phi, tp_theta)
    x2,y2,z2=get_point(phi_n, theta_n)
    return max(min(((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5, 1000), 0.0001)

def get_distance_mod2(tp_phi, tp_theta, phi_n, theta_n, regul=0.1):
    def get_point(phi, theta, regul=0.1):
        
        add_theta = regul * np.arcsin(np.sin(10*theta))
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta + add_theta)
        z = math.cos(phi)
        return (x,y,z)
    x1,y1,z1=get_point(tp_phi, tp_theta)
    x2,y2,z2=get_point(phi_n, theta_n)
    return max(min(((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5, 1000), 0.0001)

def get_adj_matrix_for_directed_rand_projection_ga(n_nodes_phi, n_nodes_theta, phi_window=10, theta_window=15,
                              range_phi=(-math.pi / 2, math.pi / 2),
                              range_theta=(- math.pi, math.pi),
                              y_min=-1, y_max=1, x_min=-1, x_max=1, regul=0.15):
    step_phi = (range_phi[1] - range_phi[0]) / n_nodes_phi
    step_theta = (range_theta[1] - range_theta[0]) / n_nodes_theta

    radius = get_radius(step_phi,step_theta, phitheta2xy_np_mod)
    y_min *= radius
    y_max *= radius
    x_min *= radius
    x_max *= radius
        
    adj_matrix = sparse.lil_matrix((n_nodes_phi*n_nodes_theta, n_nodes_phi*n_nodes_theta), dtype=np.float32)

    for phi_index in range(n_nodes_phi):
        tp_phi = phi_index * step_phi + range_phi[0]
        for theta_index in range(n_nodes_theta):
            tp_theta = theta_index * step_theta  + range_theta[0]
            tp_index = get_index(phi_index, theta_index, n_nodes_theta)

            for delta_phi in range(phi_window*2 + 1):
                delta_phi = delta_phi - phi_window
                for delta_theta in range(theta_window*2 + 1):
                    delta_theta = delta_theta - theta_window
                    if delta_phi == 0 and delta_theta == 0:
                        continue

                    if((np.abs(np.cos(tp_theta + delta_theta * step_theta)-np.cos(tp_theta)) < 0.5) and 
                        (np.abs(tp_phi + delta_phi * step_phi-tp_phi) < math.pi/2)):
                        
                        x, y = phitheta2xy_np_mod(tp_phi + delta_phi * step_phi,
                                           tp_theta + delta_theta * step_theta, tp_phi, tp_theta)
                        x *= -1
                        if y_min <= y <= y_max and x_min <= x <= x_max:
                            if (phi_index + delta_phi < n_nodes_phi) and (theta_index + delta_theta < n_nodes_theta):
                                theta_index_n = theta_index + delta_theta
                                phi_index_n = phi_index + delta_phi
                                if (theta_index_n < 0):
                                    theta_index_n += n_nodes_theta

                                # process negative
                                if (phi_index_n >= 0) and (theta_index_n >= 0):
                                    index_neighbour = get_index(phi_index_n, theta_index_n, n_nodes_theta)

                                    phi_n = phi_index_n * step_phi - math.pi / 2
                                    theta_n = theta_index_n * step_theta

                                    distance = get_distance_mod(tp_phi, tp_theta, phi_n, theta_n, regul)
                                    adj_matrix[tp_index,index_neighbour] = 1. / distance
                            
    return adj_matrix.tocoo()


def get_neighb_hor(curr_i, curr_j, off, w, h):
    """Computes the neighbour with horizontal offset for upper half of the image and 
        vertical offset for the lower part of the cubemap image."""
    if(0 <= curr_i < h): 
        if(0 <= curr_j < w):
            if ((curr_j + off) >= w):  return curr_i, curr_j + off
            elif ((curr_j + off) < 0): return 2*w+curr_j+off, 2*w-(curr_i+1)
        elif(w <= curr_j < 2*w):
            if ((curr_j + off) >= 2*w): return curr_i, curr_j + off
            elif ((curr_j + off) < w):  return curr_i, curr_j + off
        elif(2*w <= curr_j < 3*w):
            if ((curr_j + off) >= 3*w):  return h + (curr_j+off-3*w), 2*w-(curr_i+1)
            elif ((curr_j + off) < 2*w): return curr_i, curr_j + off
        return curr_i, curr_j+off
    elif(h <= curr_i < 2*h):
        if(0 <= curr_j < w):
            if ((curr_i + off) >= 2*h): return h-(curr_i+off-2*h+1), w - (curr_j+1)
            elif ((curr_i + off) < h):  return curr_i+off, 2*w + curr_j
        elif(w <= curr_j < 2*w):
            if ((curr_i + off) >= 2*h): return h - (curr_j-w+1), (curr_i + off - 2*h)
            elif ((curr_i + off) < h):  return h - (curr_j-w+1), 3*w + (curr_i+off-h)
        elif(2*w <= curr_j < 3*w):
            if ((curr_i + off) >= 2*h): return (curr_i+off-2*h), curr_j-2*w
            elif ((curr_i + off) < h):  return h-(curr_i+off+1), 3*w-(curr_j-2*w+1)
        return curr_i+off, curr_j

def get_neighb_ver(curr_i, curr_j, off, w, h):
    """Computes the neighbour with vertical offset for upper half of the image and 
        horizontal offset for the lower part of the cubemap image."""
    if(0 <= curr_i < h): 
        if(0 <= curr_j < w):
            if ((curr_i + off) >= h):  return 2*h-(curr_i+off-h+1), w-(curr_j+1)
            elif ((curr_i + off) < 0): return 2*h+(curr_i+off), 2*w+curr_j            
        elif(w <= curr_j < 2*w):
            if ((curr_i + off) >= h):  return 2*h-(curr_j-w+1), (curr_i + off-h)
            elif ((curr_i + off) < 0): return 2*h-(curr_j-w+1), 3*w + (curr_i + off)
        elif(2*w <= curr_j < 3*w):
            if ((curr_i + off) >= h):  return (curr_i+off), curr_j-2*w
            elif ((curr_i + off) < 0): return h - (curr_i+off+1), 3*w - (curr_j-2*w+1)
        return curr_i+off, curr_j
    elif(h <= curr_i < 2*h):
        if(0 <= curr_j < w):
            if ((curr_j + off) >= w):  return curr_i, curr_j + off
            elif ((curr_j + off) < 0): return w+curr_j+off, 2*w-(curr_i-w+1)
        elif(w <= curr_j < 2*w):
            if ((curr_j + off) >= 2*w): return curr_i, curr_j + off
            elif ((curr_j + off) < w):  return curr_i, curr_j + off
        elif(2*w <= curr_j < 3*w):
            if ((curr_j + off) >= 3*w):  return (curr_j+off-3*w), 2*w-(curr_i-h+1)
            elif ((curr_j + off) < 2*w): return curr_i, curr_j + off
        return curr_i, curr_j+off  
    
def get_agj_matrix_for_cubemap(h, w, shift_hor, shift_ver):
    """Computes the neighbourhood of the current point with coordinates (curr_i, curr_j)."""
    points_i = []
    points_j = []
    fsz = 1
    
    h, w = int(h / 2), int(w / 3)

    adj = sparse.lil_matrix((2*h * 3*w, 2*h * 3*w), dtype=np.float32)

    for curr_i in range(2*h):
        for curr_j in range(3*w):
            if(curr_i < h):
                n_curr_i, n_curr_j = get_neighb_hor(curr_i, curr_j, shift_hor, w, h)
            else:
                n_curr_i, n_curr_j = get_neighb_ver(curr_i, curr_j, shift_hor, w, h)
            n_curr_i = n_curr_i % (2*h)
            n_curr_j = n_curr_j % (3*w)
            if(curr_i < h):
                neigh_i, neigh_j = get_neighb_ver(n_curr_i, n_curr_j, shift_ver, w, h)
            else:
                neigh_i, neigh_j = get_neighb_hor(n_curr_i, n_curr_j, shift_ver, w, h)
            if not curr_j * 2*h + curr_i == neigh_j * 2*h + neigh_i:
                adj[curr_i * 3*w + curr_j, neigh_i * 3*w + neigh_j] = 1. / (shift_hor**2 + shift_ver**2)**0.5

    return adj.tocoo()


