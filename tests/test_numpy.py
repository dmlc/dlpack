import pytest
import numpy as np
import torch
import tensorflow as tf
import mxnet as mx
from . import dtypes, arrays, TfTensor, MxArray

@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromself(data, dtype):
    x = np.array(data, dtype=dtype)
    y = np._from_dlpack(x)
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtorch(data, dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"torch doesn't support {dtype}.")
    x = torch.tensor(data, dtype=dt)
    y = np._from_dlpack(x)
    expected_y = x.numpy()
    np.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_fromtensorflow(data, dtype):
    if 'complex' in dtype:
        pytest.xfail("tensorflow currently doesn't support complex dtypes.")
    x = tf.constant(data, dtype=dtype)
    y = np._from_dlpack(TfTensor(x))
    expected_y = x.numpy()
    np.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize('data', arrays)
@pytest.mark.parametrize('dtype', dtypes)
def test_frommxnet(data, dtype):
    try:
        x = mx.nd.array(data, dtype=dtype)
    except KeyError:
        pytest.skip(f"mxnet doesn't support {dtype}.")
    y = np._from_dlpack(MxArray(x))
    expected_y = x.asnumpy()
    np.testing.assert_array_equal(y, expected_y)


def test_byteswapped():
    dt = np.dtype('=i8').newbyteorder()
    x = np.arange(5, dtype=dt)

    with pytest.raises(TypeError):
        np._from_dlpack(x)


def test_invalid_dtype():
    x = np.asarray(np.datetime64('2021-05-27'))

    with pytest.raises(TypeError):
        np._from_dlpack(x)


def non_contiguous_testcases(x, wrapper=None, to_numpy=lambda x: x):
    if wrapper is None:
        wrapper = lambda x: x

    y1 = x[0]
    np.testing.assert_array_equal(to_numpy(y1), np._from_dlpack(wrapper(y1)))

    y2 = x[:, 0]
    np.testing.assert_array_equal(to_numpy(y2), np._from_dlpack(wrapper(y2)))

    y3 = x[1, :]
    np.testing.assert_array_equal(to_numpy(y3), np._from_dlpack(wrapper(y3)))

    y4 = x[1:3, 3:5]
    np.testing.assert_array_equal(to_numpy(y4), np._from_dlpack(wrapper(y4)))


def test_non_contiguous():
    x = np.arange(25).reshape((5, 5))
    non_contiguous_testcases(x)

    # test against torch
    x = torch.arange(25).reshape((5, 5))
    non_contiguous_testcases(x, to_numpy=lambda x: x.numpy())

    # test against tensorflow
    x = tf.constant(range(25), shape=(5, 5))
    non_contiguous_testcases(x, TfTensor, lambda x: x.numpy())

    # test against mxnet
    x = mx.nd.arange(25).reshape((5, 5))
    non_contiguous_testcases(x, MxArray, lambda x: x.asnumpy())
