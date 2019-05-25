# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import networkx as nx
import random
import time
from math import ceil
from sklearn.model_selection import train_test_split


# Normalize the matrix by row so that each row adds up to 1
def scale_sim_mat(mat):
    mat = mat - np.diag(np.diag(mat))  # Diagonal needs to be 0
    for i in range(len(mat)):
        if mat[i].sum() > 0:
            mat[i] = mat[i] / mat[i].sum()
        else:  # Handling those points that only connect to themselves
            mat[i] = 1 / (len(mat)-1)
            mat[i, i] = 0
    return mat


# Get the co-occurrence matrix by accumulating the state matrix after every hop, each random walk has a restart probability 1-alpha.
def random_surf(mat, num_hops, alpha):
    num_nodes = len(mat)
    adj_matrix = scale_sim_mat(mat)
    p0 = np.eye(num_nodes, dtype='float32')
    p = p0
    a = np.zeros((num_nodes, num_nodes), dtype='float32')
    for i in range(num_hops):
        p = (alpha * np.dot(p, adj_matrix)) + ((1 - alpha) * p0)
        a = a + p
    return a


# Compute the improved PPMI matrix of the co-occurrence matrix
def ppmi_matrix(mat):
    num_nodes = len(mat)
    mat = scale_sim_mat(mat)
    col_sum = np.sum(mat, axis=0).reshape(1, num_nodes)
    col_sum = np.power(col_sum, 0.75)  # smoothing, reduce the effect that low frequency makes high PPMI
    # row_sum all become 1 after scaling, so we don't need to divide it anymore,
    # and multiply num_nodes to make sure the PPMI values are not too small to lose precision
    ppmi = np.log(np.divide(num_nodes * mat, col_sum))
    ppmi[np.isinf(ppmi)] = 0.0
    ppmi[np.isneginf(ppmi)] = 0.0
    ppmi[ppmi < 0.0] = 0.0
    return ppmi


# Calculate Mahalanobis distance between embedded vectors
def get_pairwise_distances(embeddings, m_metric):
    dot_product = tf.matmul(tf.matmul(embeddings, m_metric), tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)
    # prevent gradient explosion
    mask = tf.to_float(tf.equal(distances, 0.0))
    distances = tf.sqrt(distances + mask * 1e-16) * (1.0 - mask)  # sqrt and refine distance
    return distances


# Compute mask tensor that satisfies: labels[i]=labels[j]!=labels[k] and i,j,k are all different
def get_triplet_mask(labels):
    index_ne = tf.logical_not(tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool))  # n*n-sized bool matrix which means indexes unequal
    label_eq = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))  # n*n-sized bool matrix which means labeles equal
    label_i_eq_j = tf.expand_dims(label_eq, 2)
    label_i_ne_k = tf.expand_dims(tf.logical_not(label_eq), 1)
    index_i_ne_j = tf.expand_dims(index_ne, 2)
    # Inner 'and' makes label_i=label_j!=label_k, so that index_i!=index_j,index_j!=index_k; outer 'and' makes index_i!=index_j
    mask = tf.logical_and(tf.logical_and(label_i_eq_j, label_i_ne_k), index_i_ne_j)
    return mask


class TEA(object):

    def __init__(self, len_feature, n_hidden=(500, 100), alpha=0.1, learning_rate=0.001):
        # the last dimension of n_hidden is the dimension of embedding，others are hidden layer sizes
        self.init_op = None
        self.train_op = None

        # global parameters during model training
        self.features = None
        self.features_noised = None
        self.labels = None
        self.loss = None

        # global parameters during model using
        self.features_new = None
        self.embeddings = None

        # Construct Graph and Session
        self.graph = tf.Graph()
        self.build(len_feature, n_hidden, alpha, learning_rate)
        self.sess = tf.Session(graph=self.graph)

    # forward one layer
    @staticmethod
    def _dense(input_layer, weight, bias, activation=None):
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    # add noise to data, randomly set zeros with Proportion of noise_rate
    @staticmethod
    def _noise(data, noise_rate):
        data[np.random.rand(data.shape[0], data.shape[1]) < noise_rate] = 0
        return data

    # Compute triplet loss, and get the average of non-zeros
    @staticmethod
    def _get_triplet_loss(labels, embeddings, margin, m_metric):
        distances = get_pairwise_distances(embeddings, m_metric)
        anchor_positive_dist = tf.expand_dims(distances, 2)
        anchor_negative_dist = tf.expand_dims(distances, 1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # Place invalid triplet with 0 and remove negative values
        triplet_loss = tf.maximum(tf.multiply(tf.to_float(get_triplet_mask(labels)), triplet_loss), 0)
        # Calculate the number of positive values and compute the average value
        num_positive_triplets = tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, 1e-15)))
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)  # to prevent divide 0 error
        return triplet_loss

    # encoder forward
    def encoder(self, features, n_hidden, activation=tf.nn.leaky_relu):
        for i, n in enumerate(n_hidden[:-1]):
            weights = tf.get_variable('encode_W{}'.format(i + 1), shape=[n, n_hidden[i + 1]], dtype=tf.float32)
            biases = tf.get_variable('encode_b{}'.format(i + 1), shape=(n_hidden[i + 1]), dtype=tf.float32)
            features = self._dense(features, weights, biases, activation)
        return features

    # decoder forward
    def decoder(self, features, n_hidden, activation=tf.nn.leaky_relu):
        for i, n in enumerate(n_hidden[:-1]):
            weights = tf.get_variable('decode_W{}'.format(i + 1), shape=[n, n_hidden[i + 1]], dtype=tf.float32)
            biases = tf.get_variable('decode_b{}'.format(i + 1), shape=(n_hidden[i + 1]), dtype=tf.float32)
            features = self._dense(features, weights, biases, activation)
        return features

    # the whole Graph
    def build(self, len_feature, n_hidden, alpha, learning_rate):
        # refine layer sizes
        n_encoder = [len_feature] + n_hidden
        n_decoder = list(reversed(n_hidden)) + [len_feature]

        with self.graph.as_default():
            # inputs
            self.features = tf.placeholder(tf.float32, shape=(None, len_feature))
            self.features_noised = tf.placeholder(tf.float32, shape=(None, len_feature))
            self.labels = tf.placeholder(tf.int32, shape=None)

            with tf.variable_scope("default"):
                initializer = tf.contrib.layers.xavier_initializer()
                tf.get_variable_scope().set_initializer(initializer)

                # Compute the reconstruct loss of autoEncoder
                embedded = self.encoder(features=self.features_noised, n_hidden=n_encoder)
                decoded = self.decoder(features=embedded, n_hidden=n_decoder)
                loss_recon = tf.reduce_mean(tf.square((self.features - decoded)))

                # weight elimination L2 regularizer, removed while debugging other parameters
                # regularizer =

                # Initialize semidefinate matrix M=L^TL
                l_metric = tf.get_variable("L_metric", [n_hidden[-1], n_hidden[-1]],
                                           initializer=tf.constant_initializer(np.eye(n_hidden[-1])))
                # compute triplet loss
                loss_triplet = self._get_triplet_loss(self.labels, embedded, 1.0,
                                                      tf.matmul(tf.transpose(l_metric), l_metric))

                # compute loss
                self.loss = loss_recon + alpha * loss_triplet

            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # optimizar

            # embedding with trained parameters
            self.features_new = tf.placeholder(tf.float32, shape=(None, len_feature))
            with tf.variable_scope("default") as scope:
                scope.reuse_variables()
                self.embeddings = self.encoder(features=self.features_new, n_hidden=n_encoder,
                                               activation=tf.nn.leaky_relu)

            self.init_op = tf.global_variables_initializer()  # initialization

    # define training process
    def fit(self, nodes, labels, epochs=10, batch_size=None, train_rate=0.7, encode_all=False, noise_rate=0.25):
        num_nodes = nodes.shape[0]
        num_labels = len(np.unique(labels))
        if not batch_size:  # divide data into 10 batch when batch_size not set
            batch_size = ceil(num_nodes / 10)
            if batch_size < num_labels * 15:  # make sure there's enough valid triplet for each batch
                batch_size = num_labels * 15
        # train test split
        index_train, index_test = train_test_split(range(num_nodes), test_size=1-train_rate, stratify=labels)  # , random_state=42
        iter_max = ceil(len(index_train) / batch_size)
        iter_max_test = ceil(len(index_test) / batch_size)

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print("Epoch %2d/%2d: " % (epoch + 1, epochs))
            start_time = time.time()

            # mini-batch
            random.shuffle(index_train)
            random.shuffle(index_test)
            loss_epoch = 0
            # train
            for i in range(iter_max):
                batch = index_train[i * batch_size: (i + 1) * batch_size]
                feed_dict = {self.features: nodes[batch],
                             self.features_noised: self._noise(nodes[batch], noise_rate=noise_rate),
                             self.labels: labels[batch]}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                # accumulate the total error of the epoch and output the weighted average error at the end
                loss_epoch += len(batch) * loss

            # test
            for i in range(iter_max_test):
                batch = index_test[i * batch_size: (i + 1) * batch_size]
                feed_dict = {self.features: nodes[batch],
                             self.features_noised: self._noise(nodes[batch], noise_rate=noise_rate),
                             self.labels: np.zeros_like(labels[batch])}  # set label same to make triplet-loss 0 so that the label information are not used
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                loss_epoch += len(batch) * loss
            print("time = %ds , avg loss = %8.4f" % (time.time() - start_time, loss_epoch / num_nodes))

        if encode_all:
            return self.encode(nodes), labels
        else:  # get the embedding and labels of test
            return self.encode(nodes[index_test]), labels[index_test]

    # encode data
    def encode(self, nodes):
        embeddings = self.sess.run(self.embeddings, feed_dict={self.features_new: nodes})
        return embeddings


if __name__ == '__main__':
    # read data
    dataSet = 'cora'
    method = 'tea50'
    print('read graph ...')
    g = nx.read_edgelist('data/' + dataSet + '_edges.txt', nodetype=int)
    # g = nx.read_edgelist('data/' + dataSet + '_edges.txt', nodetype=int, create_using=nx.DiGraph())
    X = nx.to_numpy_array(g, range(0, g.number_of_nodes()))
    y = np.zeros(shape=(X.shape[0]), dtype=int)
    with open('data/' + dataSet + '_labels.txt') as fin:
        for line in fin.readlines():
            vec = line.strip().split()
            y[int(vec[0])] = int(vec[1])

    # get co-occurrence matrix by random surfing and calculate the PPMI of it
    X = random_surf(X, 3, 0.98)
    X = ppmi_matrix(X)

    model = TEA(len_feature=g.number_of_nodes(),
                n_hidden=[520, 100],
                alpha=0.15,
                learning_rate=0.001)

    # train_rate defines the proportion of labeled data
    embedding, truth = model.fit(nodes=X,
                                 labels=y,
                                 epochs=10,
                                 batch_size=None,
                                 train_rate=0.5,
                                 encode_all=False,
                                 noise_rate=0.25)  # fixed

    np.savetxt('output/' + dataSet + '_' + method + '_emb.txt', embedding)
    np.savetxt('output/' + dataSet + '_' + method + '_lbl.txt', truth, fmt="%d")
