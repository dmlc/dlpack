import pytest
import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import mxnet as mx
from . import dtypes, arrays, TfTensor, MxArray

np_config.enable_numpy_behavior()


def tensorflow_assert_equal(x, y):
    assert tf.reduce_all(x == y)
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.shape == y.shape


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromself(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    x = tf.constant(data, dtype=dtype)
    y = tf.experimental.dlpack.from_dlpack(TfTensor(x).__dlpack__())
    tensorflow_assert_equal(x, y)


@pytest.mark.skip(reason="tensorflow crashes when importing NumPy arrays.")
@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromnumpy(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    x = np.array(data, dtype=dtype)
    y = tf.experimental.dlpack.from_dlpack(x.__dlpack__())
    expected_y = tf.constant(data, dtype=dtype)
    tensorflow_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtorch(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = torch.tensor(data, dtype=dt)
    y = tf.experimental.dlpack.from_dlpack(x.__dlpack__())
    expected_y = tf.constant(data, dtype=dtype)
    tensorflow_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_frommxnet(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    try:
        x = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = tf.experimental.dlpack.from_dlpack(MxArray(x).__dlpack__())
    expected_y = tf.constant(data, dtype=dtype)
    tensorflow_assert_equal(y, expected_y)
