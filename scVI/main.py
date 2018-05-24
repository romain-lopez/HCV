
# coding: utf-8

# loading modules
import scVI
import scVIgenqc
import IDR
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import pickle
from benchmarking import *
import numpy as np
import scipy.stats as stats
from helper import *
import time
import pandas as pd
import matplotlib.pyplot as plt


# parameters


learning_rate = 0.0004
epsilon = 0.01



# expression data
data_path = "/home/ubuntu/single-cell-scVI/data/10xPBMCs/"
expression_train = np.load(data_path + "de/data_train.npy")
expression_test = np.load(data_path + "de/data_test.npy")

# qc metrics
r_train = np.load(data_path + "design_train.npy")
r_test = np.load(data_path + "design_test.npy")
qc_train = np.load(data_path + "qc_train.npy")[:,  4:]
qc_test = np.load(data_path + "qc_test.npy")[:, 4:]


normalized_qc_train = np.copy(qc_train)
normalized_qc_train = np.log(normalized_qc_train / (1- normalized_qc_train))
mean = np.mean(normalized_qc_train, axis=0)
std = np.std(normalized_qc_train, axis=0)
normalized_qc_train -= mean
normalized_qc_train /= std

normalized_qc_test = np.copy(qc_test)
normalized_qc_test = np.log(normalized_qc_test / (1- normalized_qc_test))
normalized_qc_test -= mean
normalized_qc_test /= std



# getting priors for scVI
log_library_size = np.log(np.sum(expression_train, axis=1))
mean, var = np.mean(log_library_size), np.var(log_library_size)


# labels
c_train = np.loadtxt(data_path + "label_train")
c_test = np.loadtxt(data_path + "label_test")

# batch info
b_train = np.loadtxt(data_path + "b_train")
b_test = np.loadtxt(data_path + "b_test")

# corrupted data
X_zero, i, j, ix =         np.load(data_path + "imputation/X_zero.npy"), np.load(data_path + "imputation/i.npy"),        np.load(data_path + "imputation/j.npy"), np.load(data_path + "imputation/ix.npy")
        
#gene info
micro_array_result = pd.read_csv(data_path+"de/gene_info.csv")
gene_names = micro_array_result["ENSG"]
gene_symbols = micro_array_result["GS"]
cd_p_value = micro_array_result["CD_P.Value"]
bdc_p_value = micro_array_result["BDC_P.Value"]
bdc2_p_value = micro_array_result["BDC2_P.Value"]

print expression_train.shape, expression_test.shape, normalized_qc_train.shape, normalized_qc_test.shape


# Computational graph


tf.reset_default_graph()

expression = tf.placeholder(tf.float32, (None, expression_train.shape[1]), name='x')
qc = tf.placeholder(tf.float32, (None, qc_train.shape[1]), name="qc")

kl_scalar = tf.placeholder(tf.float32, (), name='kl_scalar')
hsic_scalar = tf.placeholder(tf.float32, (), name='hsic_scalar')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
training_phase = tf.placeholder(tf.bool, (), name='training_phase')


model = scVIgenqc.scVIGenQCModel(expression=expression, qc=qc, kl_scale=kl_scalar, hsic_scale=hsic_scalar,                           optimize_algo=optimizer, phase=training_phase,                            library_size_mean=mean, library_size_var=var, n_latent=5, n_latent_qc=2)

# Session creation
sess = tf.Session()



# start a new graph and fit the training set
sess.run(tf.global_variables_initializer())
result = train_model_qc(model, (expression_train, expression_test),                         (qc_train, qc_test), sess, 300, hsic=0)
#plot_training_info(result)


dic_train = {model.expression: expression_test, model.training_phase:True, model.kl_scale:0}
dic_train[model.qc] = qc_test
dic_train[model.hsic_scale] = 0
print sess.run([tf.reduce_mean(model.recon_x), tf.reduce_mean(model.recon_s), model.loss], feed_dict=dic_train)


# HERE FOLLOWS A SIMPLE MODIFICATION OF THE SCVI DIFFERENTIAL EXPRESSION

def sample_posterior(model, X, QC, M_z, M_u, alter_qc=True):
    # shape and simulation
    results = {}
    ind = np.arange(X.shape[0])

    # repeat the data for sampling
    X_m = np.repeat(X, M_z, axis=0)
    QC_m = np.repeat(QC, M_z, axis=0)
    ind = np.repeat(ind, M_z, axis=0)
    
        
    #NN part
    dic_x = {expression: X_m, training_phase:False, kl_scalar:1., qc:QC_m} 
    z_m, l_m, u_m = sess.run((model.z, model.library, model.u), feed_dict=dic_x)
    
    # repeat for u
    z_m_u = np.repeat(z_m, M_u, axis=0)
    l_m_u = np.repeat(l_m, M_u, axis=0)
    u_m_u = np.repeat(u_m, M_u, axis=0)
    
    if alter_qc:    
        # ignore the QC information
        u_m_u = np.random.normal(size=(z_m_u.shape[0], model.n_latent_qc))
        
    dic_z = {model.z: z_m_u, model.library:l_m_u, model.u:u_m_u, training_phase:False, kl_scalar:1.}
    
    # regenerate rates
    rate, dropout, scale = sess.run((model.px_rate, model.px_dropout, model.px_scale), feed_dict=dic_z)
    dispersion = np.tile(sess.run((tf.exp(model.px_r))), (rate.shape[0], 1))
    
    results["library"] = l_m
    results["mean"] = rate
    results["latent"] = z_m
    results["dispersion"] = dispersion
    results["dropout"] = dropout
    results["sample_rate"] = scale
    results["index"] = ind
    return results


def get_sampling(model, subset_a, subset_b, M_z, M_u, alter_qc):
    #get q(z| xa) and q(z| xb) and sample M times from it, then output gamma parametrizations
    res_a = sample_posterior(model, expression_train[subset_a], qc_train[subset_a], M_z, M_u, alter_qc=alter_qc)
    res_b = sample_posterior(model, expression_train[subset_b], qc_train[subset_b], M_z, M_u, alter_qc=alter_qc)
    return res_a, res_b



def get_statistics(res_a, res_b, M_p=10000, permutation=False):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    """
    
    #agregate dataset
    samples = np.vstack((res_a["sample_rate"], res_b["sample_rate"]))
    
    # prepare the pairs for sampling
    list_1 = list(np.arange(res_a["sample_rate"].shape[0]))
    list_2 = list(res_a["sample_rate"].shape[0] + np.arange(res_b["sample_rate"].shape[0]))
    if not permutation:
        #case1: no permutation, sample from A and then from B
        u, v = np.random.choice(list_1, size=M_p), np.random.choice(list_2, size=M_p)
    else:
        #case2: permutation, sample from A+B twice
        u, v = (np.random.choice(list_1+list_2, size=M_p),                     np.random.choice(list_1+list_2, size=M_p))
    
    # then constitutes the pairs
    first_set = samples[u]
    second_set = samples[v]
    
    res = np.mean(first_set >= second_set, 0)
    res = np.log(res) - np.log(1-res)
    return res

interest = "BDC"
couple_celltypes = (4, 0)
rank_auc = 800
p_prior = 0.25

# getting p_values
p_value = micro_array_result[interest + "_adj.P.Val"]
signed_p_value = - np.log10(p_value) * np.sign(micro_array_result[interest + "_logFC"])
# setting up parameters
A, B, M_z, M_u = 400, 400, 2, 10
set_a = np.where(c_train == couple_celltypes[0])[0]
set_b = np.where(c_train == couple_celltypes[1])[0]

# subsampling cells and computing statistics
subset_a = np.random.choice(set_a, A)
subset_b = np.random.choice(set_b, B)
res_a, res_b = get_sampling(model, subset_a, subset_b, M_z, M_u, alter_qc=True)
st = get_statistics(res_a, res_b, M_p=100000)

# de evaluation param
idr = IDR.IDR()

alter_qc=True

l = []
for i in range(20):
    print i
    subset_a = np.random.choice(set_a, A)
    subset_b = np.random.choice(set_b, B)
    res_a, res_b = get_sampling(model, subset_a, subset_b, M_z, M_u, alter_qc=alter_qc)
    st = get_statistics(res_a, res_b, M_p=100000)
    res = idr.fit(np.abs(st), -np.log(p_value), p_prior=p_prior)
    auc = auc_score_threshold(p_value, np.abs(st), rank_auc, p_value=False)
    res = list(res)
    res.append([auc])
    l.append([x[0] for x in res])    


print np.array(l)[:, 0].mean(), np.array(l)[:, 0].std()
print np.array(l)[:, 1].mean(), np.array(l)[:, 1].std()
print np.array(l)[:, 2].mean(), np.array(l)[:, 2].std()



