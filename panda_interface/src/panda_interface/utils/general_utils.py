import numpy as np
from contextlib import contextmanager
import heapq
import inspect
import time
import copy
import re
import random
import functools
from functools import partial, reduce
import collections
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg


class AverageTimer(AverageMeter):
    """Times whatever is inside the with self.time(): ... block, exposes average etc like AverageMeter."""
    @contextmanager
    def time(self):
        self.start = time.time()
        yield
        self.update(time.time() - self.start)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


def dict_concat(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        raise ValueError("Dict keys are not equal. got {} vs {}.".format(d1.keys(), d2.keys()))
    for key in d1:
        d1[key] = np.concatenate((d1[key], d2[key]))

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


def subdict(dict, keys, strict=True):
    if not strict:
        keys = dict.keys() & keys
    return AttrDict((k, dict[k]) for k in keys)

def concat_inputs(*inp):
    """ Concatenates tensors together. Used if the tensors need to be passed to a neural network as input. """
    max_n_dims = np.max([len(tensor.shape) for tensor in inp])
    inp = torch.cat([add_n_dims(tensor, max_n_dims - len(tensor.shape)) for tensor in inp], dim=1)
    return inp


def select_e_0_e_g(seq, start_ind, end_ind):
    e_0 = batchwise_index(seq, start_ind)
    e_g = batchwise_index(seq, end_ind)
    return e_0, e_g


def shuffle_with_seed(arr, seed=1):
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(arr)
    return arr


def interleave_lists(*args):
    """Interleaves N lists of equal length."""
    for l in args:
        assert len(l) == len(args[0])      # all lists need to have equal length
    return [val for tup in zip(*args) for val in tup]


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})

def get_end_ind(pad_mask):
    """
    :param pad_mask: torch tensor with 1 where there is an actual image and zeros where there's padding
    pad_mask has shape batch_size x max_seq_len
    :return:
    """
    max_seq_len = pad_mask.shape[1]
    end_ind = torch.argmax(pad_mask * torch.arange(max_seq_len, dtype=torch.float, device=pad_mask.device), 1)

    return end_ind

class HasParameters:
    def __init__(self, **kwargs):
        self.build_params(kwargs)

    def build_params(self, inputs):
        # If params undefined define params
        try:
            self.params
        except AttributeError:
            self.params = self.get_default_params()
            self.params.update(inputs)

def maybe_retrieve(d, key):
    if hasattr(d, key):
        return d[key]
    else:
        return None


class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self



class DictFlattener:
    """Flattens all elements in ordered dict into single vector, remembers structure and can unflatten back."""
    def __init__(self):
        self._example_struct = None

    def __call__(self, d):
        """Flattens dict d into vector."""
        assert isinstance(d, OrderedDict)
        if self._example_struct is None:
            self._example_struct = copy.deepcopy(d)
        assert d.keys() == self._example_struct.keys()
        return np.concatenate([d[key] for key in d])

    def unflatten(self, v):
        """Restores original dict structure."""
        output, idx = OrderedDict(), 0
        for key in self._example_struct:
            output[key] = v[idx : idx + self._example_struct[key].shape[0]]
            idx += self._example_struct[key].shape[0]
        return output


def pretty_print(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key) + ':')
            pretty_print(value, indent+1)
        else:
            print('\t' * indent + str(key) + ':' + '\t' + str(value))
