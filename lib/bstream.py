import numpy as np
import tensorflow as tf

"""Simulating sequences with Tensorflow models."""


def modelsfunc(model, qpm_, nstream, nhist1, nhist2):
    """
    Create a tensorflow compiled function to run simulation for a model.
    :param model: tensorflow model.
    :param qpm_: Bucket parameters from getqpm(rate, bdepth).
    :param nstream: Number of parallel streams this functoin will be called on.
    :param nhist1: Number of historical offloading metric values the model needs.
    :param nhist2: Number of historical inter-arrival times the model needs
                   (When negative, the model needs no inter-arrival time input).
    :return: a function that you should call as func(tf.data.Dataset, int) to get average loss,
            where the second parameter to the function is number of elements in the dataset.

    You can use the data2tf to get a tuple corresponding to these arguments:

    simfunc = modelsfunc(...)
    tfdset = data2tf(...)

    avg_loss = simfunc(*tfdset)
    """
    qpm = tf.constant(qpm_)
    idx = tf.range(nstream)
    intnum = tf.constant(nhist2 + 1)
    # Create tensorflow variables to store the total reward and the current state.
    tot_gain = tf.Variable(0, trainable=False, dtype=tf.float64)
    nstate = tf.Variable([qpm_[2]] * nstream, trainable=False, dtype=tf.int32)
    mhist = tf.Variable(np.zeros((nstream, nhist1 + 1)), trainable=False, dtype=tf.float32)
    ihist = tf.Variable(np.zeros((nstream, max(nhist2 + 1, 1))), trainable=False, dtype=tf.int32)

    @tf.function
    def simloop(dataset):
        for (_m, _r, _i) in dataset:
            # Update the the history window of offloading metrics and inter-arrival time.
            mhist.assign(tf.concat((tf.reshape(_m, (-1, 1)), mhist[:, :-1]), 1))
            qipt = tf.concat((mhist, tf.cast(ihist, tf.float32)[:, :intnum]), 1)
            ihist.assign(tf.concat((tf.reshape(_i, (-1, 1)), ihist[:, :-1]), 1))
            qvals = model(qipt)
            # Determine the offloading decision with the predicted Q-values, and update token bucket state.
            deci = (qvals[:, (qpm[1] - qpm[0]):(qpm[2] - qpm[0] + 1)] <= qvals[:, (qpm[2] - qpm[0] + 1):])
            ifsend = tf.gather_nd(deci, tf.stack([idx, tf.maximum(tf.constant(0), nstate - qpm[1])], 1))
            ifsend = tf.logical_and(ifsend, nstate >= qpm[1])
            nstate.assign(tf.minimum(qpm[2], nstate - tf.where(ifsend, qpm[1], 0) + qpm[0] * _i))
            # Update the cumulative offloading reward.
            igain = tf.reduce_sum(tf.cast(tf.where(ifsend, _r, 0), tf.float64))
            tot_gain.assign_add(igain)

    def mstream(dataset, sz_=1.0):
        # Clear the variables and run the simulation loop.
        tot_gain.assign(np.float64(0))
        nstate.assign([qpm_[2]] * nstream)
        mhist.assign(np.zeros((nstream, nhist1 + 1), np.float32))
        ihist.assign(np.zeros((nstream, max(nhist2 + 1, 1)), np.int32))
        simloop(dataset)
        return tot_gain.numpy() / sz_

    return mstream


def data2tf(dset, iset, metrics, rewards):
    """Convert dataset from numpy arrays to a (tf.Dataset, int:size) tuple."""
    metrics, rewards = metrics.astype(np.float32), rewards.astype(np.float32)
    metrics, rewards = metrics[dset.T], rewards[dset.T]
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(metrics),
                                   tf.data.Dataset.from_tensor_slices(rewards),
                                   tf.data.Dataset.from_tensor_slices(iset.T)))
    return dataset, np.size(dset)
