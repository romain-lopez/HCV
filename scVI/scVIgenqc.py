"""
Code for the paper single-cell Variational Inference (scVI) paper

Romain Lopez, Jeffrey Regier, Michael Cole, Michael Jordan, Nir Yosef
EECS, UC Berkeley

"""

import functools
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from scVI import * 
from scipy.special import gamma


def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                  name='clipped_logit')
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)


def hsic_objective(z, s):
    
    # use a gaussian RBF for every variable
    def K(x1, x2, gamma=1.): 
        dist_table = tf.expand_dims(x1, 0) - tf.expand_dims(x2, 1)
        return tf.transpose(tf.exp(-gamma * tf.reduce_sum(dist_table **2, axis=2)))
      
    d_z = z.get_shape().as_list()[1]
    d_s = s.get_shape().as_list()[1]
    
    gz = 2 * gamma(0.5 * (d_z+1)) / gamma(0.5 * d_z)
    gs = 2 * gamma(0.5 * (d_z+1)) / gamma(0.5 * d_z)
    
    zz = K(z, z, gamma= 1. / (2. * gz))
    ss = K(s, s, gamma= 1. / (2. * gs))
        
        
    hsic = 0
    hsic += tf.reduce_mean(zz * ss) 
    hsic += tf.reduce_mean(zz) * tf.reduce_mean(ss)
    hsic -= 2 * tf.reduce_mean( tf.reduce_mean(zz, axis=1) * tf.reduce_mean(ss, axis=1) )
    return tf.sqrt(hsic)


class scVIGenQCModel:

    def __init__(self, expression=None, qc=None, kl_scale=None, hsic_scale=None, phase=None,\
                 library_size_mean = None, library_size_var = None, \
                 dispersion="gene", n_layers=1, n_hidden=128, n_latent=10, n_latent_qc=3, \
                 dropout_rate=0.1, optimize_algo=None, zi=True):
        """
        Main parametrization of the scVI algorithm.

        Notes and disclaimer:
        + We recommend to put kl_scale to 1 for every tasks except clustering where 0 will lead better discrepency between the clusters
        + Applying a too harsh penalty will ruin your biology info. We recommend using less than a 100. From ongoing tests, using zero actually removes batch effects as well as the paper results.
        + We recommend the dispersion parameter to be gene specific (or batch-specific as in the paper) as in ZINB-WaVE if you do not have enough cells
        + To better remove library size effects between clusters, mention the log-library size prior for each batch (like in the paper)


        Variables:
        expression: tensorflow variable of shape (minibatch_size x genes), placeholder for input counts data
        batch_ind: tensorflow variable for batch indices (minibatch_size) with integers from 0 to n_batches - 1
        kl_scale: tensorflow variable for scalar multiplier of the z kl divergence
        hsic_scale: tensorflow variable for scalar multiplier of the HSIC penalty
        phase: tensorflow variable for training phase
        library_size_mean = either a number or a list for each batch of the mean log library size
        library_size_var = either a number or a list for each batch of the variance of the log library size
        dispersion: "gene" (n_genes params) or "gene-batch" (n_genes x n_batches params) or "gene-cell" (a neural nets)
        n_layers: a integer for the number of layers in each neural net. We use 1 throughout the paper except on the 1M dataset where we tried (1, 2, 3) hidden layers
        n_hidden: number of neurons for each hidden layer. Always 128.
        n_latent: number of desired dimension for the latent space
        dropout_rate: rate to use for the dropout layer (see elementary layer function). always 0.1
        optimize_algo: a tensorflow optimizer
        zi: whether to use a ZINB or a NB distribution
        """
        
        # Gene expression placeholder
        if expression is None:
            raise ValueError("provide a tensor for expression data")
        self.expression = expression
        
        print("Running scVI on "+ str(self.expression.get_shape().as_list()[1]) + " genes")
        
        # batch correction
        if qc is None:
            raise ValueError("provide a tensor for quality control data")
        self.qc = qc
        self.hsic_scale = hsic_scale

        print("Got " + str(self.qc.get_shape().as_list()[1]) + "QCs in the data")
        print("Will apply a HSIC penalty")
        
        #kl divergence scalar
        if kl_scale is None:
            raise ValueError("provide a tensor for kl scalar")
        self.kl_scale = kl_scale
                
        #prior placeholder
        if library_size_mean is None or library_size_var is None:
            raise ValueError("provide prior for library size")
            
        if type(library_size_mean) in [float, np.float64] :
            self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
            self.library_size_var = tf.to_float(tf.constant(library_size_var))
            
        else:
            raise ValueError("provide correct prior for library size")
                
        
        # high level model parameters
        if dispersion not in ["gene", "gene-batch", "gene-cell"]:
            raise ValueError("dispersion should be in gene / gene-batch / gene-cell")
        self.dispersion = dispersion
        
        print("Will work on mode " + self.dispersion + " for modeling inverse dispersion param")
        
        self.zi = zi
        if zi:
            print("Will apply zero inflation")
        
        # neural nets architecture
        self.n_hidden = n_hidden
        self.n_latent_qc = n_latent_qc
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_input = self.expression.get_shape().as_list()[1]
        self.n_input_qc = self.qc.get_shape().as_list()[1]
        

        print(str(self.n_layers) + " hidden layers at " + str(self.n_hidden) + " each for a final " + str(self.n_latent) + " latent space and " + str(self.n_latent_qc) + "dim for qc")
        
        # on training variables
        self.dropout_rate = dropout_rate
        if phase is None:
            raise ValueError("provide an optimization metadata (phase)")
        self.training_phase = phase
        if optimize_algo is None:
            raise ValueError("provide an optimization method")
        self.optimize_algo = optimize_algo
        
        # call functions
        self.variational_distribution
        self.sampling_latent
        self.generative_model
        self.optimize
        self.optimize_test
        self.imputation

    @define_scope
    def variational_distribution(self):
        """
        defines the variational distribution or inference network of the model
        q(z, l, u | x, s) = q(z|x) q(u | z, s) q(l | x)


        """

        #q(z | x)
        x = tf.log(1 + self.expression)
        h = dense(x, self.n_hidden, activation=tf.nn.relu, \
                    bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        for layer in range(2, self.n_layers + 1):
            h = dense(h, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        
        self.qz_m = dense(h, self.n_latent, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        self.qz_v = dense(h, self.n_latent, activation=tf.exp, \
                bn=False, keep_prob=None, phase=self.training_phase)
        
        # q(l | x)
        h = dense(x, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        self.ql_m = dense(h, 1, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        self.ql_v = dense(h, 1, activation=tf.exp, \
                bn=False, keep_prob=None, phase=self.training_phase)
        
         # q(u | x, s)
        x = tf.log(1 + self.expression)
        y = tf.log(1e-8 + self.qc) - tf.log(1 - self.qc + 1e-8)
        h = tf.concat([x, y], 1)
        h = dense(h, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        self.qu_m = dense(h, self.n_latent_qc, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        self.qu_v = dense(h, self.n_latent_qc, activation=tf.exp, \
                bn=False, keep_prob=None, phase=self.training_phase)       

    
    @define_scope
    def sampling_latent(self):
        """
        defines the sampling process on the latent space given the var distribution
        """
            
        self.z = gaussian_sample(self.qz_m, self.qz_v)
        self.library = gaussian_sample(self.ql_m, self.ql_v)
        self.u = gaussian_sample(self.qu_m, self.qu_v)

    @define_scope
    def generative_model(self):
        """
        defines the generative process given a latent variable (the conditional distribution)
        """
        
        # p(s | u)
        h_s = dense(self.u, 25,
                  activation=tf.nn.relu, bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        
        #self.ps_m = dense(h_s, self.n_input_qc, activation=None, \
        #        bn=False, keep_prob=None, phase=self.training_phase)
        #self.ps_v = dense(h_s, self.n_input_qc, activation=tf.exp, \
        #        bn=False, keep_prob=None, phase=self.training_phase) 
        
        self.ps_logit = dense(h_s, self.n_input_qc, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        
        
        # p(x | z, u)
        h = tf.concat([self.z, self.u], 1)
        h = dense(h, self.n_hidden,
                  activation=tf.nn.relu, bn=True, keep_prob=None, phase=self.training_phase)
                
        for layer in range(2, self.n_layers + 1):
            h = dense(h, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
            
        h = tf.concat([h, self.u], 1)
        
        #mean gamma
        self.px_scale = dense(h, self.n_input, activation=tf.nn.softmax, \
                    bn=False, keep_prob=None, phase=self.training_phase)
        
        #dispersion
        if self.dispersion == "gene-cell":
            self.px_r = dense(h, self.n_input, activation=None, \
                    bn=False, keep_prob=None, phase=self.training_phase)
        elif self.dispersion == "gene":
            self.px_r = tf.Variable(tf.random_normal([self.n_input]), name="r")
        else:
            raise ValueError("wrong value for dispersion")

            
        #mean poisson
        self.px_rate = tf.exp(self.library) * self.px_scale

        #dropout
        if self.zi:
            self.px_dropout = dense(h_s, self.n_input, activation=None, \
                    bn=False, keep_prob=None, phase=self.training_phase)
        

    @define_scope
    def optimize(self):
        """
        write down the loss and the optimizer
        """            
        local_l_mean = self.library_size_mean
        local_l_var = self.library_size_var
        local_dispersion = tf.exp(self.px_r)
        
        # VAE loss
        if self.zi:
            self.recon_x = log_zinb_positive(self.expression, self.px_rate, local_dispersion, \
                                  self.px_dropout)
        else:
            self.recon_x = log_nb_positive(self.expression, self.px_rate, local_dispersion)
            
        #self.recon_s = log_normal(self.qc, self.ps_m, self.ps_v)
        self.recon_s = log_bernoulli_with_logits(self.qc, self.ps_logit)

        self.kl_gauss_z = 0.5 * tf.reduce_sum(\
                        tf.square(self.qz_m) + self.qz_v - tf.log(1e-8 + self.qz_v) - 1, 1)
        
        self.kl_gauss_u = 0.5 * tf.reduce_sum(\
                        tf.square(self.qu_m) + self.qu_v - tf.log(1e-8 + self.qu_v) - 1, 1)
        
        self.kl_gauss_l = 0.5 * tf.reduce_sum(\
                        tf.square(self.ql_m - local_l_mean) / local_l_var  \
                            + self.ql_v / local_l_var \
                            + tf.log(1e-8 + local_l_var)  - tf.log(1e-8 + self.ql_v) - 1, 1)
        
        self.ELBO_gau = tf.reduce_mean(self.recon_x + self.recon_s - self.kl_scale * self.kl_gauss_z - self.kl_gauss_l - self.kl_gauss_u)
        
        self.hsic = hsic_objective(self.z, self.u)
        self.loss = - self.ELBO_gau + self.hsic_scale * self.hsic
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = self.optimize_algo
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)
    
    @define_scope
    def optimize_test(self):
        # Test time optimizer to compare log-likelihood score of ZINB-WaVE
        update_ops_test = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "variational")
        test_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "variational")
        optimizer_test = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.1)
        with tf.control_dependencies(update_ops_test):
            self.test_step = optimizer_test.minimize(self.loss, var_list=test_vars)
    
    @define_scope
    def imputation(self):
        # more information of zero probabilities
        if self.zi:
            self.zero_prob = tf.nn.softplus(- self.px_dropout + tf.exp(self.px_r) * self.px_r - tf.exp(self.px_r) \
                             * tf.log(tf.exp(self.px_r) + self.px_rate + 1e-8)) \
                             - tf.nn.softplus( - self.px_dropout)
            self.dropout_prob = - tf.nn.softplus( - self.px_dropout)