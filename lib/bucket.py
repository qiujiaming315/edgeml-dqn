import numpy as np

"""Functions for dealing with token buckets."""


def getqpm(rate, bdepth, maxp=100):
    """Return the scaled token bucket parameters (q,p,m) (integer) such that q/p ~ rate, m/p ~ bdepth."""
    denom = np.arange(maxp, dtype=np.int64) + 1
    rerr, berr = denom * rate, denom * bdepth
    err = (rerr - np.floor(rerr) + berr - np.floor(berr)) / denom
    _p = denom[np.argmin(err)]
    _q = np.int64(np.floor(rate * _p))
    _m = np.int64(np.floor(bdepth * _p))
    gcd = np.gcd(np.gcd(_q, _p), _m)
    return _q // gcd, _p // gcd, _m // gcd


def getvpidx(rate, bdepth):
    """Get the token numbers for indices of value and policy vectors."""
    qpm = getqpm(rate, bdepth)
    vidx = np.arange(qpm[0], qpm[2] + 1, dtype=np.float64) / qpm[1]
    pidx = np.arange(qpm[1], qpm[2] + 1, dtype=np.float64) / qpm[1]
    return vidx, pidx


def qflatlen(qpm):
    """Return size of vector to represent Q values of all [n,a]."""
    return 2 * (qpm[2] + 1) - qpm[0] - qpm[1]


def qflat2dec(qpm, qflat):
    """Turn Q value matrix into a decision matrix."""
    qflat = qflat[:, (qpm[1] - qpm[0]):].reshape((-1, 2, qpm[2] + 1 - qpm[1]))
    return (qflat[:, 1, :] >= qflat[:, 0, :]).astype(np.uint8)


def qflatprev(qpm, qprime, rew, discount, trans):
    """Compute Q[n,a] = ar + max_a' Q[n'=(n.a), a']."""
    a0len = qpm[2] + 1 - qpm[0]
    qmax = qprime[:, :a0len]
    qmax[:, (qpm[1] - qpm[0]):] = np.maximum(qmax[:, (qpm[1] - qpm[0]):], qprime[:, a0len:])

    if len(trans.shape) > 1:
        qtrans = np.concatenate((trans[qpm[0]:, :], trans[:(a0len - qpm[1] + qpm[0]), :]), axis=0)
        qtrans = np.transpose(qtrans)
        qtgt = np.matmul(qmax, qtrans)
    else:
        qtgt = np.zeros_like(qprime)
        trans = trans.astype(np.int32)
        idx1 = qpm[0] * trans[:, np.newaxis] + np.arange(a0len)
        idx1 = np.minimum(idx1, a0len - 1)
        idx2 = qpm[0] * (trans - 1)[:, np.newaxis] + np.arange(a0len - qpm[1] + qpm[0])
        idx2 = np.minimum(idx2, a0len - 1)
        qtgt[:, :a0len] = qmax[np.arange(qprime.shape[0])[:, np.newaxis], idx1]
        qtgt[:, a0len:] = qmax[np.arange(qprime.shape[0])[:, np.newaxis], idx2]
    qnew = discount * qtgt
    qnew[:, a0len:] += rew[:, np.newaxis]
    return qnew


def iidpolicy(qpm, metrics, rewards, trans, discount=0.9999, itparam=(1e4, 1e-6)):
    """
    Find optimal policy thresholds assuming iid offloading metrics and periodic image arrival.
    :param qpm: Bucket parameters from getqpm(rate, bdepth).
    :param metrics: Training set of offloading metrics.
    :param rewards: Training set of offloading rewards.
    :param trans: The transition matrix for token bucket states.
    :param discount: Discount factor to apply (default 0.9999).
    :return: MDP policy thresholds.
    """

    # Sort metrics and compute F(theta) and G(theta)
    idx = np.argsort(-metrics)
    metrics, rewards = np.float64(metrics[idx]), np.float64(rewards[idx])
    gtheta = np.cumsum(rewards) / len(rewards)
    ftheta = np.float64(np.arange(1, len(rewards) + 1)) / len(rewards)
    fgt = (np.reshape(ftheta, (-1, 1)), np.reshape(gtheta, (-1, 1)))

    thresh = (np.amax(metrics) - np.amin(metrics)) * itparam[1]

    # Do value iterations
    value, policy = np.zeros((qpm[2] - qpm[0] + 1), np.float64), None
    for i in range(int(itparam[0])):
        vprev, pprev = value.copy(), policy

        # If n < P/P, can't send
        value[:(qpm[1] - qpm[0])] = discount * np.dot(trans[qpm[0]:qpm[1]], vprev)

        # If n >= P/P:
        vnosend = np.dot(trans[qpm[1]:], vprev)
        vsend = np.dot(trans[:(qpm[2] - qpm[1] + 1)], vprev)
        score = fgt[1] + discount * (fgt[0] * vsend + (1 - fgt[0]) * vnosend)
        value[(qpm[1] - qpm[0]):] = np.amax(score, 0)
        policy = metrics[np.argmax(score, 0)]

        if i > 0:
            if np.max(np.abs(policy - pprev)) < thresh:
                break

    return policy
