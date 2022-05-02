import pytest
import numpy as np
import torch
import tensorflow as tf
import mxnet as mx
from . import dtypes, arrays, TfTensor, MxArray


def mxnet_assert_equal(x, y):
    x, y = x.asnumpy(), y.asnumpy()
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromself(data, dtype):
    try:
        x = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = mx.nd.from_dlpack(MxArray(x).__dlpack__())
    mxnet_assert_equal(x, y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromnumpy(data, dtype):
    x = np.array(data, dtype=dtype)
    try:
        expected_y = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = mx.nd.from_dlpack(x.__dlpack__())
    mxnet_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtorch(data, dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = torch.tensor(data, dtype=dt)
    try:
        expected_y = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = mx.nd.from_dlpack(x.__dlpack__())
    mxnet_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtensorflow(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    x = tf.constant(data, dtype=dtype)
    try:
        expected_y = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = mx.nd.from_dlpack(TfTensor(x).__dlpack__())
    mxnet_assert_equal(y, expected_y)
