from dataclasses import dataclass
import numpy as np

"""Offloading metric and inter-arrival time sequence Generators."""


######################## Functions for generating offloading metric sequences. ########################


def seq_iid(baselen, outlen, outnum, seed=0, opt=None):
    """
    Generate offloading metric sequences (represented by image indexes) with iid sampling.
    :param baselen: Dataset size to generate sequences from.
    :param outlen: Length of each output sequence.
    :param outnum: Number of generated sequences.
    :param seed: Random number seed for repeatablilty.
    :param opt: Options.
    :return: an outnum x outlen array with entries between 0 and baselen-1 (image index).
    """
    rstate = np.random.RandomState(seed)
    out = rstate.randint(0, baselen, size=(outnum, outlen), dtype=np.int32)
    return out


@dataclass
class LocResetOpt:
    """Options for the local reset model."""
    spread: float = 0.1  # Ratio of training set to repeat from.
    rprob: float = 0.01  # Probably of resetting.


_LROPT = LocResetOpt()


def seq_locreset(baselen, outlen, outnum, seed=0, opt=_LROPT):
    """
    Generate offloading metric sequences (represented by image indexes) based on the local reset model.
    :param baselen: Dataset size to generate sequences from.
    :param outlen: Length of each output sequence.
    :param outnum: Number of generated sequences.
    :param seed: Random number seed for repeatablilty.
    :param opt: Options (LocResetOpt object with members spread, rprob).
    :return: an outnum x outlen array with entries between 0 and baselen-1 (image index).
    """
    baselen, outlen, outnum = int(baselen), int(outlen), int(outnum)
    rstate = np.random.RandomState(seed)
    # Randomly select the amount of deviation from the location (delta).
    delta = rstate.randint(0, int(opt.spread * baselen) + 1, (outnum, outlen), dtype=np.int32)
    # Randomly select the indexes for location resetting.
    rprob = rstate.random_sample((outnum, outlen - 1))
    _y, _x = np.where(rprob <= opt.rprob)
    dset = np.zeros_like(delta)
    # Randomly pick sampling locations.
    dset[:, 0] = rstate.randint(0, baselen, (outnum,), dtype=np.int32)
    dset[_y, _x + 1] = rstate.randint(0, baselen, (len(_y),), dtype=np.int32)
    # Add the deviation to the location.
    dset = np.cumsum(dset, 1)
    dset = np.mod(dset + delta, baselen)
    return dset


######################## Functions for generating inter-arrival time sequences. ########################


def _geomatrix(qpm, aprob):
    """Utility function to generate token bucket state transition matrix for geometric distribution."""
    rnum, cnum = qpm[2] + 1, qpm[2] - qpm[0] + 1
    pnum = qpm[2] // qpm[0]
    m = np.zeros((rnum, cnum))
    pidx = np.arange(rnum)[:, np.newaxis] + np.arange(pnum) * qpm[0]
    pidx = np.minimum(pidx, cnum - 1)
    gprob = np.power(1 - aprob, np.arange(pnum)) * aprob
    for i in range(pnum):
        m[np.arange(rnum), pidx[:, i]] += gprob[i]
    m[:, -1] += 1 - np.sum(gprob)
    return m


def _spmatrix(qpm, intlen):
    """Utility function to generate token bucket state transition matrix through sample distribution."""
    rnum, cnum = qpm[2] + 1, qpm[2] - qpm[0] + 1
    pnum = qpm[2] // qpm[0]
    m = np.zeros((rnum, cnum))
    pidx = np.arange(rnum)[:, np.newaxis] + np.arange(pnum) * qpm[0]
    pidx = np.minimum(pidx, cnum - 1)
    cumfreq = 0
    for i in range(pnum):
        freq = np.sum(intlen == i + 1) / np.size(intlen)
        m[np.arange(rnum), pidx[:, i]] += freq
        cumfreq += freq
    m[:, -1] += 1 - cumfreq
    return m


def int_periodic(outlen, outnum, seed=0, opt=None):
    """
    Generate inter-arrival time sequences based on the periodic arrival model.
    :param outlen: Length of each output sequence.
    :param outnum: Number of generated sequences.
    :param seed: Random number seed for repeatablilty.
    :param opt: Options.
    :return: an all-one outnum x outlen array, the average inter-arrival time, and a transition matrix generator.
    """
    outlen, outnum = int(outlen), int(outnum)
    intlen = np.ones((outnum, outlen), np.int32)

    def trans_matrix(qpm):
        return _geomatrix(qpm, 1)

    return intlen, (1, trans_matrix)


@dataclass
class DoublePeriodicOpt:
    """Options for the Markov-modulated periodic arrival model."""
    int1: int = 1  # Inter-arrival time in state 0.
    int2: int = 3  # Inter-arrival time in state 1.
    tprob1: float = 0.001  # Probability of transition from state 0 to 1.
    tprob2: float = 0.0005  # Probability of transition from state 1 to 0.


_DPOPT = DoublePeriodicOpt()


def int_doubleperiodic(outlen, outnum, seed=0, opt=_DPOPT):
    """
    Generate inter-arrival time sequences based on the 2-state Markov-modulated periodic arrival model.
    :param outlen: Length of each output sequence.
    :param outnum: Number of generated sequences.
    :param seed: Random number seed for repeatablilty.
    :param opt: Options (DoublePeriodicOpt object with member minint1, maxint1, minint2, maxint2, tprob1, tprob2).
    :return: an outnum x outlen array, the average inter-arrival time, and a transition matrix generator.
    """
    outlen, outnum = int(outlen), int(outnum)
    rstate = np.random.RandomState(seed)
    # Compute some statistics (e.g., state probability) based on the parameters.
    pi1 = opt.tprob2 / (opt.tprob1 + opt.tprob2)
    avgnum = 1 / opt.tprob1 / opt.int1 + 1 / opt.tprob2 / opt.int2
    tmplen = max(1, int(outlen / avgnum / 3))
    # Declare variables that keep track of the generated sequences.
    intlen = np.zeros((outnum, outlen + 1), np.int32)
    inits = rstate.rand(outnum) < pi1
    curidx = np.zeros((outnum), np.int32)
    curstm = np.ones((outnum), np.bool)
    # Generate new segments for sequences shorter than the specified length.
    while np.any(curstm):
        for i in np.arange(outnum)[curstm]:
            # Use a new seed for random numbers to avoid repeated segments.
            rstate.seed(seed)
            seed += 1
            # Randomly select the lengths of each state.
            sts1 = rstate.geometric(opt.tprob1, size=tmplen)
            sts2 = rstate.geometric(opt.tprob2, size=tmplen)
            maxnum = max(np.amax(sts1) // opt.int1, np.amax(sts2) // opt.int2)
            uint1 = np.ones((tmplen, maxnum), dtype=np.int32) * opt.int1
            uint2 = np.ones((tmplen, maxnum), dtype=np.int32) * opt.int2
            sts = np.zeros((2 * tmplen), np.int32)
            uint = np.zeros((2 * tmplen, maxnum), np.int32)
            # Select which state to start with based on state probability.
            if inits[i]:
                sts[::2], sts[1::2] = sts1, sts2
                uint[::2], uint[1::2] = uint1, uint2
            else:
                sts[::2], sts[1::2] = sts2, sts1
                uint[::2], uint[1::2] = uint2, uint1
            # Retrieve the sequence of inter-arrival times based on how long each state lasts.
            uint = np.cumsum(uint, axis=1)
            bmask = uint <= sts[:, np.newaxis]
            offset = np.cumsum(sts)
            offset[1:] = offset[:-1]
            offset[0] = 0
            uint += offset[:, np.newaxis]
            ints = uint[bmask]
            ints[1:] -= ints[:-1]
            intnum = np.size(ints)
            # Check if the generated sequence is sufficiently long.
            enum = outlen - curidx[i] + 1
            curstm[i] = intnum < enum
            fnum = min(enum, intnum)
            # Append the newly generated sub-sequences to the end of the generated sequences.
            intlen[i, curidx[i]:curidx[i] + fnum] = ints[:fnum]
            curidx[i] += fnum

    intlen = intlen[:, 1:]

    def trans_matrix(qpm):
        return _spmatrix(qpm, intlen)

    return intlen, (np.mean(intlen), trans_matrix)


SEQFUNCS = [seq_iid, seq_locreset]
INTFUNCS = [int_periodic, int_doubleperiodic]
