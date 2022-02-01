Welcome to dlpack's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

DLPack is a in-memory data buffer to exchange data between different array/tensor representation libraries.

DLPack consists of a header file that defines an intermedieate protocol to represent data in-memory.
This protocol can be utilied by other array/tensor libraries to convert from their representation to an intermedieate
representation and vice versa. This create a bridge between different array libraries and enables the user to easily
integrate multiple such libraries in their projects.

Many Python array libraries have functions/utilities to convert to and from a dlpack tensor. Below is an example of how one
can convert a PyTorch tensor to NumPy array:

.. code-block:: Python

    import numpy as np
    import torch

    # PyTorch tensor
    torch_tensor = torch.arange(10, device='cpu')

    # Convert the PyTorch tensor to NumPy array
    np_array = np._from_dlpack(torch_tensor)


C API
=====

.. doxygendefine:: DLPACK_EXTERN_C

.. doxygendefine:: DLPACK_VERSION

.. doxygendefine:: DLPACK_DLL

.. doxygenenum:: DLDeviceType

.. doxygenstruct:: DLDevice
   :members:

.. doxygenenum:: DLDataTypeCode

.. doxygenstruct:: DLDataType
   :members:

.. doxygenstruct:: DLTensor
   :members:

.. doxygenstruct:: DLManagedTensor
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
