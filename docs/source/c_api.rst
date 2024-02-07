.. _c_api:

C API (``dlpack.h``)
====================

Macros
~~~~~~

.. doxygendefine:: DLPACK_EXTERN_C

.. doxygendefine:: DLPACK_MAJOR_VERSION

.. doxygendefine:: DLPACK_MINOR_VERSION

.. doxygendefine:: DLPACK_DLL

.. doxygendefine:: DLPACK_FLAG_BITMASK_READ_ONLY

.. doxygendefine:: DLPACK_FLAG_BITMASK_IS_COPIED

Enumerations
~~~~~~~~~~~~

.. doxygenenum:: DLDeviceType

.. doxygenenum:: DLDataTypeCode

Structs
~~~~~~~

.. doxygenstruct:: DLPackVersion
   :members:

.. doxygenstruct:: DLDevice
   :members:

.. doxygenstruct:: DLDataType
   :members:

.. doxygenstruct:: DLTensor
   :members:

.. doxygenstruct:: DLManagedTensor
   :members:

.. doxygenstruct:: DLManagedTensorVersioned
   :members:
