import ctypes
import numpy as np
from .dlpack import _c_str_dltensor, DLManagedTensor


class _Holder:
    """A wrapper that combines a pycapsule and array_interface for consumption by  numpy.

    Parameters
    ----------
    array_interface : dict
        A description of the underlying memory.

    pycapsule : PyCapsule
        A wrapper around the dlpack tensor that will be converted to numpy.
    """

    def __init__(self, array_interface, pycapsule) -> None:
        self.__array_interface__ = array_interface
        self._pycapsule = pycapsule


def to_numpy(pycapsule) -> np.ndarray:
    """Convert a dlpack tensor into a numpy array without copying.

    Parameters
    ----------
    pycapsule : PyCapsule
        A pycapsule wrapping a dlpack tensor that will be converted.

    Returns
    -------
    np_array : np.ndarray
        A new numpy array that uses the same underlying memory as the input
        pycapsule.
    """
    pycapsule = ctypes.py_object(pycapsule)
    assert ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor)
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
        pycapsule, _c_str_dltensor
    )
    dl_managed_tensor = ctypes.cast(dl_managed_tensor, ctypes.POINTER(DLManagedTensor))
    holder = _Holder(dl_managed_tensor.contents.__array_interface__, pycapsule)
    return np.ctypeslib.as_array(holder)
