import tensorflow as tf
import mxnet as mx

dtypes = [
    'uint8', 'uint16', 'uint32', 'uint64',
    'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128'
]

arrays = [
    [1, 2, 3],    # 1D array
    1,            # ndim = 0 array
    [],           # empty array
    [[1, 2, 3],
     [4, 5, 6]],  # multi-dimensional array
]


class TfTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=0):
        return tf.experimental.dlpack.to_dlpack(self.tensor)

    def __dlpack_device__(self):
        return (1, 0)  # we only test CPU tensors for now.


class MxArray:
    def __init__(self, array):
        self.array = array

    def __dlpack__(self, stream=0):
        return mx.nd.to_dlpack_for_read(self.array)

    def __dlpack_device__(self):
        return (1, 0)
