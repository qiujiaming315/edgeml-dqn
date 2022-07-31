import argparse
import numpy as np
import tensorflow as tf

import lib.utils as ut
import lib.data as dt
import lib.genseq as gs
import lib.bstream as bst
import lib.bucket as bkt
from lib.model import mlp, iid2model

"""Script to simulate the trained DQN on test sequences."""


def main(opts):
    """Main function."""
    # Load data and sort images according to the offloading metrics.
    (tr_m, tr_r), (ts_m, ts_r) = dt.load_base(opts.npz)
    tr_idx, ts_idx = np.argsort(tr_m), np.argsort(ts_m)
    tr_m, tr_r = tr_m[tr_idx], tr_r[tr_idx]
    ts_m, ts_r = ts_m[ts_idx], ts_r[ts_idx]

    # Generate the test sequences.
    _tsz = (int(1e5), 100)  # 100 test sequences with 100,000 images each.
    # Test sequences of offloading metrics.
    seqfunc = gs.SEQFUNCS[opts.stype]
    _LROPT = gs.LocResetOpt(opts.spread, opts.rprob)
    seq_opt = [None, _LROPT]
    tset = seqfunc(len(ts_m), _tsz[0], _tsz[1], seed=0, opt=seq_opt[opts.stype])
    # Test sequences of inter-arrival times.
    intfunc = gs.INTFUNCS[opts.itype]
    _DPOPT = gs.DoublePeriodicOpt(opts.int1, opts.int2, opts.tprob1, opts.tprob2)
    int_opt = [None, _DPOPT]
    tint, (tau, trans_matrix) = intfunc(_tsz[0], _tsz[1], seed=0, opt=int_opt[opts.itype])
    simt_tf = bst.data2tf(tset, tint, ts_m, ts_r)
    # Set the number of history inter-arrival time properly.
    opts.nhist2 = max(opts.nhist2, -1)
    # Prepare the token bucket parameters.
    qpm = bkt.getqpm(opts.tr, opts.tb)
    flatlen = bkt.qflatlen(qpm)

    # Compute the average loss achieved by each policy on the test set.
    # Average loss of the weak and strong classifiers.
    npz_data = np.load(opts.npz)
    wloss, sloss = np.mean(npz_data['wcost_ts'][tset]), np.mean(npz_data['scost_ts'][tset])
    # Average loss of the Lower Bound policy.
    threshold = tr_m[np.int(len(tr_m) * (1 - opts.tr * tau)) - 1]
    lb_reward = np.sum(ts_r[ts_m > threshold]) / np.size(ts_r)
    lb_loss = wloss - lb_reward
    # Average loss of the Baseline policy.
    bl_policy = np.ones((qpm[2] - qpm[1] + 1), np.float64) * threshold
    basefunc = bst.modelsfunc(iid2model(qpm, bl_policy), qpm, _tsz[1], 0, -1)
    bl_reward = basefunc(*simt_tf)
    bl_loss = wloss - bl_reward
    # Average loss of the MDP policy.
    iid_save = np.load(opts.wts + '/iid.npz')
    mdp_policy = iid_save["policy"]
    iidsfunc = bst.modelsfunc(iid2model(qpm, mdp_policy), qpm, _tsz[1], 0, -1)
    mdp_reward = iidsfunc(*simt_tf)
    mdp_loss = wloss - mdp_reward
    # Average loss of the DQN policy.
    model = mlp(opts.nhist1 + opts.nhist2 + 2, flatlen, opts.nhidden, opts.nlayers)
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())
    ut.loadopt(opts.wts + '/opt.npz', opt, model)
    ut.loadmodel(opts.wts + '/model.npz', model)
    dqnsfunc = bst.modelsfunc(model, qpm, _tsz[1], opts.nhist1, opts.nhist2)
    dqn_reward = dqnsfunc(*simt_tf)
    dqn_loss = wloss - dqn_reward
    # Save the simulation results.
    ldata = {'weak': wloss, 'strong': sloss, 'lower_bound': lb_loss, 'baseline': bl_loss, 'mdp': mdp_loss,
             'dqn': dqn_loss}
    np.savez(opts.save + "/result.npz", **ldata)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('npz', help='Path to npz file with raw data.')
    args.add_argument('wts', help='Directory with the pre-trained DQN weights.')
    args.add_argument('save', help='Directory to save results.')

    flags = [['tr', float, 0.1, "Token rate."],
             ['tb', float, 5.0, "Token bucket depth."],
             ['stype', int, 0, "Offloading metric sequence type, 0 for i.i.d., 1 for location reset model."],
             ['itype', int, 0, "Inter-arrival time sequence type, 0 for periodic arrival, " +
              "1 for 2-state Markov modulated peirodic arrival."],

             ['nhist1', int, 95, "Number of previous metrics in state."],
             ['nhist2', int, -1, "Number of previous inter-arrival times in state. If negative, state includes no " +
              "inter-arrival times, DQN learns from transition matrix instead."],

             ['nlayers', int, 5, "Number of hidden layers in MLP."],
             ['nhidden', int, 64, "Number of units in each hidden layer."],

             ['spread', float, 0.1, "Ratio of training set to repeat from."],
             ['rprob', float, 0.01, "Probablility of resetting."],
             ['int1', int, 1, "Inter-arrival time in state 0."],
             ['int2', int, 3, "Inter-arrival time in state 1."],
             ['tprob1', float, 0.001, "Probablility of transition from state 0 to 1."],
             ['tprob2', float, 0.0005, "Probablility of transition from state 1 to 0."]]

    for _f in flags:
        args.add_argument('--' + _f[0], type=_f[1], default=_f[2], help=_f[3] + f" Default {_f[2]}.")

    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
