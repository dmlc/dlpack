import pytest
import numpy as np
import torch
import tensorflow as tf
import mxnet as mx
from . import dtypes, arrays, TfTensor, MxArray


def torch_assert_equal(x, y):
    assert torch.all(x == y)
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.shape == y.shape
    assert x.stride() == y.stride()


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromself(data, dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = torch.tensor(data, dtype=dt)
    y = torch.from_dlpack(x)
    torch_assert_equal(x, y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromnumpy(data, dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = np.array(data, dtype=dtype)
    y = torch.from_dlpack(x)
    expected_y = torch.tensor(data, dtype=dt)
    torch_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtensorflow(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = tf.constant(data, dtype=dtype)
    y = torch.from_dlpack(TfTensor(x))
    expected_y = torch.tensor(data, dtype=dt)
    torch_assert_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_frommxnet(data, dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    try:
        x = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = torch.from_dlpack(MxArray(x))
    expected_y = torch.tensor(data, dtype=dt)
    torch_assert_equal(y, expected_y)
