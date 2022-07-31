import numpy as np

"""Utility functions for handling data."""


def load_base(fname):
    """
    Load individual image data from npz file.
    :param fname: Name of npz file.
    :return: (train_metrics, train_rewards), (test_metrics, test_rewards).
    """
    # Load the offloading metrics for the training and test sets.
    npz = np.load(fname)
    tr_m, ts_m = npz['metric_tr'], npz['metric_ts']
    # Normalize the offloading metrics.
    _mu, _sd = np.mean(tr_m), np.std(tr_m)
    tr_m = (tr_m - _mu) / _sd
    ts_m = (ts_m - _mu) / _sd
    # Compute the offloading reward based on the costs.
    tr_r = npz['wcost_tr'] - npz['scost_tr']
    ts_r = npz['wcost_ts'] - npz['scost_ts']
    return (tr_m, tr_r), (ts_m, ts_r)


class DataTuples:
    """Sample tuples (segments) from a dataset."""

    def __init__(self, dset, iset, metrics, rewards, nhist1, nhist2, ntuples):
        self.metrics, self.rewards = metrics, rewards
        self.nhist1, self.nhist2 = nhist1, nhist2
        self.ntuples = ntuples
        self.nseq, self.lseq = np.shape(dset)
        self.dset, self.iset = dset.flatten(), iset.flatten()

    def sample(self):
        """Sample n tuples of (curm, reward, nextm)."""
        # Randomly select n indexes from the sequences as the end points of the segments.
        idx0 = np.random.randint(0, self.nseq, size=(self.ntuples,))
        idx1 = np.random.randint(0, self.lseq - 1, size=(self.ntuples,))
        # Retrieve the offloading rewards.
        reward = self.rewards[self.dset[idx0 * self.lseq + idx1]]
        # Retrieve the offloading metrics for the current and the next states.
        idxh = idx1[:, np.newaxis] - np.arange(self.nhist1 + 1)
        curm = self.dset[np.maximum(0, idx0[:, np.newaxis] * self.lseq + idxh)]
        curm = self.metrics[curm]
        curm[idxh < 0] = 0
        nextm = self.metrics[self.dset[idx0 * self.lseq + idx1 + 1]][:, np.newaxis]
        nextm = np.concatenate((nextm, curm[:, :-1]), -1)
        # Retrieve the inter-arrival times for the current and the next states.
        if self.nhist2 >= 0:
            idxg = idx1[:, np.newaxis] - np.arange(self.nhist2 + 1) - 1
            curi = self.iset[np.maximum(0, idx0[:, np.newaxis] * self.lseq + idxg)]
            curi[idxg < 0] = 0
            curm = np.concatenate((curm, curi), axis=1)
            nexti = self.iset[idx0 * self.lseq + idx1][:, np.newaxis]
            # Concatenate the inter-arrival times to the offloading metrics.
            nexti = np.concatenate((nexti, curi[:, :-1]), -1)
            nextm = np.concatenate((nextm, nexti), axis=1)
        return curm, reward, nextm
