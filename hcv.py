# Tensorflow implementation of HSIC
# Refers to original implementations
# https://github.com/kacperChwialkowski/HSIC
# https://cran.r-project.org/web/packages/dHSIC/index.html


from scipy.special import gamma
import tensorflow as tf
import numpy as np

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel, 
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)
    
def K(x1, x2, gamma=1.): 
    dist_table = tf.expand_dims(x1, 0) - tf.expand_dims(x2, 1)
    return tf.transpose(tf.exp(-gamma * tf.reduce_sum(dist_table **2, axis=2)))

def hsic(z, s):
    
    # use a gaussian RBF for every variable
      
    d_z = z.get_shape().as_list()[1]
    d_s = s.get_shape().as_list()[1]
    
    zz = K(z, z, gamma= bandwidth(d_z))
    ss = K(s, s, gamma= bandwidth(d_s))
        
        
    hsic = 0
    hsic += tf.reduce_mean(zz * ss) 
    hsic += tf.reduce_mean(zz) * tf.reduce_mean(ss)
    hsic -= 2 * tf.reduce_mean( tf.reduce_mean(zz, axis=1) * tf.reduce_mean(ss, axis=1) )
    return tf.sqrt(hsic)
    
# dHSIC
# list_variables has to be a list of tensorflow tensors
# for i, z_j in enumerate(list_variables):
#     k_j = K(z_j, z_j, gamma=bandwidth(z_j.get_shape().as_list()[1]))
#     if i == 0:
#         term1 = k_j
#         term2 = tf.reduce_mean(k_j)
#         term3 = tf.reduce_mean(k_j, axis=0)
#     else:
#         term1 = term1 * k_j
#         term2 = term2 * tf.reduce_mean(k_j)
#         term3 = term3 * tf.reduce_mean(k_j, axis=0)  
# dhsic = tf.sqrt(tf.reduce_mean(term1) + term2 - 2 * tf.reduce_mean(term3))
