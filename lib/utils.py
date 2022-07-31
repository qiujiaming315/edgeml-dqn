import sys
import time
import signal
import numpy as np

"""Various utility functions."""


def log(data, numit=None):
    """Log output in standard format."""
    if numit is None:
        lstr = data
    else:
        dstr = [(k + (' = %.3e' % data[k])) for k in data.keys()]
        lstr = '[%06d] ' % numit + ' '.join(dstr)
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + lstr + "\n")
    sys.stdout.flush()


def getstop():
    """Returns stop so that stop[0] is True if ctrl+c was hit."""
    stop = [False]
    _orig = [None]

    def handler(_a, _b):
        del _a
        del _b
        stop[0] = True
        signal.signal(signal.SIGINT, _orig[0])

    _orig[0] = signal.signal(signal.SIGINT, handler)

    return stop


def saveopt(fname, opt, iters):
    """Save optimizer state + niters to file."""
    weights = opt.get_weights()
    npz = {('%d' % i): weights[i] for i in range(len(weights))}
    npz['iters'] = np.int64(iters)
    np.savez(fname, **npz)


def savemodel(fname, model):
    """Save model weights to file."""
    weights = model.get_weights()
    npz = {('%d' % i): weights[i] for i in range(len(weights))}
    np.savez(fname, **npz)


def loadmodel(fname, model):
    """Restore model weights from file."""
    npz = np.load(fname)
    weights = [npz['%d' % i] for i in range(len(npz.files))]
    model.set_weights(weights)


def loadopt(fname, opt, model):
    """Restore optimizer state from file."""
    npz = np.load(fname)
    weights = [npz['%d' % i] for i in range(len(npz.files) - 1)]
    opt._create_all_weights(model.trainable_variables)
    opt.set_weights(weights)
    return npz['iters']
