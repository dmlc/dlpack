.. _c_api:

C API (``dlpack.h``)
====================

Macros
~~~~~~

.. doxygendefine:: DLPACK_EXTERN_C

.. doxygendefine:: DLPACK_VERSION

.. doxygendefine:: DLPACK_ABI_VERSION

.. doxygendefine:: DLPACK_DLL

Enumerations
~~~~~~~~~~~~

.. doxygenenum:: DLDeviceType

.. doxygenenum:: DLDataTypeCode

Structs
~~~~~~~

.. doxygenstruct:: DLDevice
   :members:

.. doxygenstruct:: DLDataType
   :members:

.. doxygenstruct:: DLPackVersion
   :members:

.. doxygenstruct:: DLTensorVersioned
   :members:

.. doxygenstruct:: DLManagedTensorVersioned
   :members:

ABI v1 Structs
~~~~~~~~~~~~~~

DLTensor and DLManagedTensor don't contain any field to export version info.
Since ABI version 2, structs DLTensorVersioned and DLManagedTensorVersioned
have been added with version info and should be used instead.

.. doxygenstruct:: DLTensor
   :members:

.. doxygenstruct:: DLManagedTensor
   :members:
