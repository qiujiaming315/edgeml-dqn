import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

import lib.utils as ut
import lib.bucket as bkt
import lib.data as dt
import lib.genseq as gs
import lib.bstream as bst
from lib.model import mlp, iid2model

"""Script to train a DQN."""


def main(opts):
    """Main function."""
    stop = ut.getstop()

    # Prepare datasets.
    ut.log("Preparing Data.")
    _tsz, _simsz = (int(1e6), 100), (int(1e5), 5)
    metrics, rewards = dt.load_base(opts.npz)[0]
    idx = np.argsort(metrics)
    metrics, rewards = metrics[idx], rewards[idx]
    # Generate sequences of offloading metric for training and validation.
    seqfunc = gs.SEQFUNCS[opts.stype]
    _LROPT = gs.LocResetOpt(opts.spread, opts.rprob)
    seq_opt = [None, _LROPT]
    tset = seqfunc(len(metrics), _tsz[0], _tsz[1], seed=0, opt=seq_opt[opts.stype])
    simset_v = seqfunc(len(metrics), _simsz[0], _simsz[1], seed=1, opt=seq_opt[opts.stype])
    simset_t = tset[:_simsz[1], :_simsz[0]].copy()
    # Generate sequences of image inter-arrival time for training and validation.
    intfunc = gs.INTFUNCS[opts.itype]
    _DPOPT = gs.DoublePeriodicOpt(opts.int1, opts.int2, opts.tprob1, opts.tprob2)
    int_opt = [None, _DPOPT]
    tint, (tau, trans_matrix) = intfunc(_tsz[0], _tsz[1], seed=0, opt=int_opt[opts.itype])
    simint_v, _ = intfunc(_simsz[0], _simsz[1], seed=1, opt=int_opt[opts.itype])
    simint_t = tint[:_simsz[1], :_simsz[0]].copy()

    # Create sampler to randomly sample state tuples during training.
    opts.nhist2 = max(opts.nhist2, -1)
    sampler = dt.DataTuples(tset, tint, metrics, rewards, opts.nhist1, opts.nhist2, opts.gbsz * opts.ngdup)
    # Create tf.data.Dataset's for running simulations to compute reward.
    simt_tf = bst.data2tf(simset_t, simint_t, metrics, rewards)
    simv_tf = bst.data2tf(simset_v, simint_v, metrics, rewards)
    # Prepare the token bucket parameters.
    qpm = bkt.getqpm(opts.tr, opts.tb)
    flatlen = bkt.qflatlen(qpm)
    tmatrix = trans_matrix(qpm)

    # Get MDP baseline results.
    Path(opts.wts).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(opts.wts + '/iid.npz'):
        iidrs = np.load(opts.wts + '/iid.npz')
        iidr_t, iidr_v = iidrs['iidr_t'], iidrs['iidr_v']
        ut.log("Loaded saved iid baseline numbers.")
    else:
        ut.log("Getting iid baseline numbers.")
        policy = bkt.iidpolicy(qpm, metrics, rewards, tmatrix, opts.discount)
        iidsfunc = bst.modelsfunc(iid2model(qpm, policy), qpm, _simsz[1], 0, -1)
        iidr_t = iidsfunc(*simt_tf)
        iidr_v = iidsfunc(*simv_tf)
        iidrs = {'iidr_t': iidr_t, 'iidr_v': iidr_v, 'policy': policy}
        np.savez(opts.wts + '/iid.npz', **iidrs)

    # Create Model & Optimizer
    model = mlp(opts.nhist1 + opts.nhist2 + 2, flatlen, opts.nhidden, opts.nlayers)
    _lr = np.power(10.0, -opts.lr)
    opt = tf.keras.optimizers.Adam(_lr)
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())
    # Create a tf.function version of bucket simulator with model.
    msfunc = bst.modelsfunc(model, qpm, _simsz[1], opts.nhist1, opts.nhist2)

    # Restore checkpoint if any.
    iters = 0
    if os.path.isfile(opts.wts + '/opt.npz'):
        iters = ut.loadopt(opts.wts + '/opt.npz', opt, model)
        ut.log("Restored optimizer.")
    if os.path.isfile(opts.wts + '/model.npz'):
        ut.loadmodel(opts.wts + '/model.npz', model)
        ut.log("Restored model.")

    # Main training loop.
    ut.log("Starting iterations at %d " % iters)
    while iters <= opts.maxiter:
        # Run simulations to compute reward at regular intervals.
        if (iters % opts.sfreq) == 0 or iters == opts.maxiter:
            modr_t = msfunc(*simt_tf)
            modr_v = msfunc(*simv_tf)
            ut.log({'iid.reward.t': iidr_t, 'iid.reward.v': iidr_v, 'reward.t': modr_t, 'reward.v': modr_v}, iters)
        if iters == opts.maxiter:
            break
        # Sample batches, compute targets, and update model.
        cur, rew, nxt = sampler.sample()
        qnxt = model.predict(nxt)
        trans = nxt[:, opts.nhist1 + 1] if opts.nhist2 >= 0 else tmatrix
        qtarget = bkt.qflatprev(qpm, qnxt, rew, opts.discount, trans)
        qtarget = qtarget - np.mean(qtarget)
        loss = model.fit(cur, qtarget, batch_size=opts.gbsz, verbose=0, shuffle=False)
        loss = np.mean(loss.history['loss'])
        ut.log({'lr': _lr, 'loss.t': loss}, iters)
        iters = iters + 1

        if stop[0]:
            break

    # Save model and optimizer state.
    if iters > 0:
        ut.log("Saving model and optimizer.")
        ut.saveopt(opts.wts + '/opt.npz', opt, iters)
        ut.savemodel(opts.wts + '/model.npz', model)
    ut.log("Stopping!")


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('npz', help='Path to npz file with raw data.')
    args.add_argument('wts', help='Directory to save weights.')

    flags = [['tr', float, 0.1, "Token rate."],
             ['tb', float, 5.0, "Token bucket depth."],
             ['stype', int, 0, "Offloading metric sequence type, 0 for i.i.d., 1 for location reset model."],
             ['itype', int, 0, "Inter-arrival time sequence type, 0 for periodic arrival, " +
              "1 for 2-state Markov modulated periodic arrival."],

             ['nhist1', int, 95, "Number of previous metrics in state."],
             ['nhist2', int, -1, "Number of previous inter-arrival times in state. If negative, state includes no " +
              "inter-arrival times, DQN learns from the state transition matrix instead."],
             ['discount', float, 0.9999, "Reward discount factor."],
             ['ngdup', int, 128, "Number of gradient updates in each Q iteration."],
             ['gbsz', int, 128, "Batch size for each gradient update."],

             ['nlayers', int, 5, "Number of hidden layers in MLP."],
             ['nhidden', int, 64, "Number of units in each hidden layer."],
             ['lr', float, 3, "-log10(Learning rate), so 3 -> 10^-3."],
             ['maxiter', int, 1000, "Maximum number of Q iterations."],
             ['sfreq', int, 100, "How frequently, in # Q iterations, to run simulations to compute reward."],

             ['spread', float, 0.1, "Ratio of training set to repeat from."],
             ['rprob', float, 0.01, "Probability of resetting."],
             ['int1', int, 1, "Image inter-arrival time in state 0."],
             ['int2', int, 3, "Image inter-arrival time in state 1."],
             ['tprob1', float, 0.001, "Probability of transition from state 0 to 1."],
             ['tprob2', float, 0.0005, "Probability of transition from state 1 to 0."]]

    for _f in flags:
        args.add_argument('--' + _f[0], type=_f[1], default=_f[2], help=_f[3] + f" Default {_f[2]}.")

    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
