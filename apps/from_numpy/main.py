from __future__ import print_function

import numpy as np
import gc
import ctypes

libmain = ctypes.cdll.LoadLibrary("./libmain.so")

class DLContext(ctypes.Structure):
  _fields_ = [("device_type", ctypes.c_int),
              ("device_id", ctypes.c_int)]

class DLDataType(ctypes.Structure):
  _fields_ = [("type_code", ctypes.c_uint8),
              ("bits", ctypes.c_uint8),
              ("lanes", ctypes.c_uint16)]
  TYPE_MAP = {
    "bool": (1, 1, 1),
    "int32": (0, 32, 1),
    "int64": (0, 64, 1),
    "uint32": (1, 32, 1),
    "uint64": (1, 64, 1),
    "float32": (2, 32, 1),
    "float64": (2, 64, 1),
  }

class DLTensor(ctypes.Structure):
  _fields_ = [("data", ctypes.c_void_p),
              ("ctx", DLContext),
              ("ndim", ctypes.c_int),
              ("dtype", DLDataType),
              ("shape", ctypes.POINTER(ctypes.c_int64)),
              ("strides", ctypes.POINTER(ctypes.c_int64)),
              ("byte_offset", ctypes.c_uint64)]

class DLManagedTensor(ctypes.Structure):
  pass

DLManagedTensorHandle = ctypes.POINTER(DLManagedTensor)

DeleterFunc = ctypes.CFUNCTYPE(None, DLManagedTensorHandle)

DLManagedTensor._fields_ = [("dl_tensor", DLTensor),
                            ("manager_ctx", ctypes.c_void_p),
                            ("deleter", DeleterFunc)]

def display(array):
  print("data =", hex(array.ctypes.data_as(ctypes.c_void_p).value))
  print("dtype =", array.dtype)
  print("ndim =", array.ndim)
  print("shape =", array.shape)
  print("strides =", array.strides)

def make_manager_ctx(obj):
  pyobj = ctypes.py_object(obj)
  void_p = ctypes.c_void_p.from_buffer(pyobj)
  ctypes.pythonapi.Py_IncRef(pyobj)
  return void_p

# N.B.: In practice, one should ensure that this function
# is not destructed before the numpy array is destructed.
@DeleterFunc
def dl_managed_tensor_deleter(dl_managed_tensor_handle):
  void_p = dl_managed_tensor_handle.contents.manager_ctx
  pyobj = ctypes.cast(void_p, ctypes.py_object)
  print("Deleting manager_ctx:")
  display(pyobj.value)
  ctypes.pythonapi.Py_DecRef(pyobj)
  print("Deleter self...")
  libmain.FreeHandle()
  print("Done")

def make_dl_tensor(array):
  # You may check array.flags here, e.g. array.flags['C_CONTIGUOUS']
  dl_tensor = DLTensor()
  dl_tensor.data = array.ctypes.data_as(ctypes.c_void_p)
  dl_tensor.ctx = DLContext(1, 0)
  dl_tensor.ndim = array.ndim
  dl_tensor.dtype = DLDataType.TYPE_MAP[str(array.dtype)]
  # For 0-dim ndarrays, strides and shape will be NULL
  dl_tensor.shape = array.ctypes.shape_as(ctypes.c_int64)
  dl_tensor.strides = array.ctypes.strides_as(ctypes.c_int64)
  for i in range(array.ndim):
    dl_tensor.strides[i] //= array.itemsize
  dl_tensor.byte_offset = 0
  return dl_tensor

def main():
  array = np.random.rand(3, 1, 30).astype("float32")
  print("Created:")
  display(array)
  c_obj = DLManagedTensor()
  c_obj.dl_tensor = make_dl_tensor(array)
  c_obj.manager_ctx = make_manager_ctx(array)
  c_obj.deleter = dl_managed_tensor_deleter
  print("-------------------------")
  del array
  gc.collect()
  libmain.Give(c_obj)
  print("-------------------------")
  del c_obj
  gc.collect()
  libmain.Finalize()
  print("-------------------------")

if __name__ == "__main__":
  main()
