.. _python-spec:

Python Specification for DLPack
===============================

The Python specification for DLPack is a part of the
`Python array API standard <https://data-apis.org/array-api/latest/index.html>`_.
More details about the spec can be found under the :ref:`data-interchange` page.


Syntax for data interchange with DLPack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The array API will offer the following syntax for data interchange:

1. A ``from_dlpack(x)`` function, which accepts (array) objects with a
   ``__dlpack__`` method and uses that method to construct a new array
   containing the data from ``x``.
2. ``__dlpack__(self, stream: int = None, version: int = None)``,
   ``__dlpack_info__(self)``, and ``__dlpack_device__(self)`` methods
   on the array object, which will be called from within ``from_dlpack``,
   to access the data, to get the maximum supported DLPack version, and
   to query what device the array is on (may be needed to pass in the
   correct stream, e.g. in the case of multiple GPUs).


Semantics
~~~~~~~~~

DLPack describes the memory layout of strided, n-dimensional arrays.
When a user calls ``y = from_dlpack(x)``, the library implementing ``x`` (the
"producer") will provide access to the data from ``x`` to the library
containing ``from_dlpack`` (the "consumer"). If possible, this must be
zero-copy (i.e. ``y`` will be a *view* on ``x``). If not possible, that library
may make a copy of the data. In both cases:

- The producer keeps owning the memory
- ``y`` may or may not be a view, therefore the user must keep the recommendation to
  avoid mutating ``y`` in mind - see :ref:`copyview-mutability`.
- Both ``x`` and ``y`` may continue to be used just like arrays created in other ways.

If an array that is accessed via the interchange protocol lives on a
device that the requesting library does not support, it is recommended to
raise a ``TypeError``.

Stream handling through the ``stream`` keyword applies to CUDA and ROCm (perhaps
to other devices that have a stream concept as well, however those haven't been
considered in detail). The consumer must pass the stream it will use to the
producer; the producer must synchronize or wait on the stream when necessary.
In the common case of the default stream being used, synchronization will be
unnecessary so asynchronous execution is enabled.

A DLPack version can be requested by passing the ``version`` keyword. The
consumer should call the ``__dlpack_info__`` method to get the maximum
DLPack version supported by the producer and request for a version both
support e.g. ``min(producer_version, consumer_version)``. If the consumer
doesn't support any version below the producer's maximum version, a
``BufferError`` should be raised. Similarly, If the producer doesn't
support the requested version, it should raise a ``BufferError``.


Implementation
~~~~~~~~~~~~~~

*Note that while this API standard largely tries to avoid discussing
implementation details, some discussion and requirements are needed
here because data interchange requires coordination between
implementers on, e.g., memory management.*

.. image:: /_static/images/DLPack_diagram.png
  :alt: Diagram of DLPack structs

*DLPack diagram. Dark blue are the structs it defines, light blue
struct members, gray text enum values of supported devices and data
types.*

The ``__dlpack__`` method will produce a ``PyCapsule`` containing a
``DLManagedTensorVersioned`` (or a ``DLManagedTensor``) that is
compatible with the DLPack and DLPack ABI version requested by the
consumer. It will be consumed immediately within ``from_dlpack`` -
therefore it is consumed exactly once, and it will not be visible
to users of the Python API.

The producer must set the ``PyCapsule`` name to ``"dltensor"`` so
that it can be inspected by name, and set ``PyCapsule_Destructor``
that calls the ``deleter`` of the ``DLManagedTensorVersioned`` (or
``DLManagedTensor``) when the ``"dltensor"``-named capsule is no
longer needed.

The consumer must transfer ownership of the ``DLManangedTensorVersioned``
(or ``DLManangedTensor``) from the capsule to its own object. It does so
by renaming the capsule to ``"used_dltensor"`` to ensure that
``PyCapsule_Destructor`` will not get called (ensured if
``PyCapsule_Destructor`` calls ``deleter`` only for capsules whose name
is ``"dltensor"``), but the ``deleter`` of the
``DLManagedTensorVersioned`` (or ``DLManagedTensor``) will be called by
the destructor of the consumer library object created to own the
``DLManagerTensorVersioned`` (or ``DLManagedTensor``) obtained from the
capsule. Below is an example of the capsule deleter written in the Python
C API which is called either when the refcount on the capsule named
``"dltensor"`` reaches zero or the consumer decides to deallocate its
array:

.. code-block:: C

   static void dlpack_capsule_deleter(PyObject *self){
      if (PyCapsule_IsValid(self, "used_dltensor")) {
         return; /* Do nothing if the capsule has been consumed. */
      }

      /* an exception may be in-flight, we must save it in case we create another one */
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);

      DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)PyCapsule_GetPointer(self, "dltensor");
      if (managed == NULL) {
         PyErr_WriteUnraisable(self);
         goto done;
      }
      /* the spec says the deleter can be NULL if there is no way for the caller to provide a reasonable destructor. */
      if (managed->deleter) {
         managed->deleter(managed);
         /* TODO: is the deleter allowed to set a python exception? */
         assert(!PyErr_Occurred());
      }

   done:
      PyErr_Restore(type, value, traceback);
   }

Note: the capsule names ``"dltensor"`` and ``"used_dltensor"`` must be
statically allocated.

When the ``strides`` field in the ``DLTensorVersioned`` (or ``DLTensor``)
struct is ``NULL``, it indicates a row-major compact array. If the array
is of size zero, the data pointer in ``DLTensorVersioned`` (or
``DLTensor``) should be set to either ``NULL`` or ``0``.

For further details on DLPack design and how to implement support for it,
refer to https://github.com/dmlc/dlpack. For details on ABI compatibility
and to upgrade to the new ABI (version 2), refer to :ref:`future-abi-compat`.

.. warning::
   DLPack contains a ``device_id``, which will be the device
   ID (an integer, ``0, 1, ...``) which the producer library uses. In
   practice this will likely be the same numbering as that of the
   consumer, however that is not guaranteed. Depending on the hardware
   type, it may be possible for the consumer library implementation to
   look up the actual device from the pointer to the data - this is
   possible for example for CUDA device pointers.

   It is recommended that implementers of this array API consider and document
   whether the ``.device`` attribute of the array returned from ``from_dlpack`` is
   guaranteed to be in a certain order or not.

.. _future-abi-compat:

Future ABI Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

ABI version 1 did not provide any fields in the structs ``DLTensor`` or
``DLManagedTensor`` to export version info. Two equivalent structs,
``DLTensorVersioned`` and ``DLManagedTensorVersioned``, have been added
since ABI version 2 (DLPack version 0.7.0) and have a ``version`` field
that can be used to export version info and check if the producer's
DLPack version is compatible with the consumer's DLPack version. This
section gives a path for Python libraries to upgrade to the new ABI
(while preserving support for the old ABI):

* ``__dlpack__`` should accept a ``version`` (int) keyword which is set to
  ``None`` by default. Consumers can use this kwarg to request certain DLPack
  versions. If ``version=None`` or ``version=60`` is requested:

  * a capsule named ``"dltensor"`` which uses the old ABI (``DLTensor`` and
    ``DLManagedTensor``) should be returned (if the producer wants to keep
    supporting it) or
  * a ``BufferError`` should be raised (if the producer doesn't want to keep
    support for the old ABI)

  Otherwise, a capsule named ``"dltensor"`` which uses the new ABI
  (``DLTensorVersioned`` and ``DLManagedTensorVersioned``) should be returned.
  If the requested version is not supported, ``__dlpack__`` should raise a
  ``BufferError``.
* Producers should implement a ``__dlpack_info__`` method that returns the
  maximum supported DLPack version. If this method does not exist, the consumer
  must use the old ABI.
* Consumers should call the ``__dlpack_info__`` method to get the maximum DLPack
  version supported by the producer. The consumer should then request a DLPack
  version (by passing the ``version`` kwarg to the ``__dlpack__`` method) that
  both support e.g. ``min(producer_version, consumer_version)`` or raise a
  ``BufferError`` if no compatible version exist. If the ``__dlpack_info__``
  method can't be found (if the method doesn't exist), the consumer should
  fallback to the old API i.e. passing no version keyword to the ``__dlpack__``
  method and expecting a capsule pointing to a ``DLManagedTensor`` (old ABI).

Reference Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

Several Python libraries have adopted this standard using Python C API, C++, Cython,
ctypes, cffi, etc:

* NumPy: `Python C API <https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/dlpack.c>`__
* CuPy: `Cython <https://github.com/cupy/cupy/blob/master/cupy/_core/dlpack.pyx>`__
* Tensorflow: `C++ <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/dlpack.cc>`__,
  `Python wrapper using Python C API <https://github.com/tensorflow/tensorflow/blob/a97b01a4ff009ed84a571c138837130a311e74a7/tensorflow/python/tfe_wrapper.cc#L1562>`__,
  `XLA <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/dlpack.cc>`__
* PyTorch: `C++ <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/DLConvertor.cpp>`__,
  `Python wrapper using Python C API <https://github.com/pytorch/pytorch/blob/c22b8a42e6038ed2f6a161114cf3d8faac3f6e9a/torch/csrc/Module.cpp#L355>`__
* MXNet: `ctypes <https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/dlpack.py>`__
* TVM: `ctypes <https://github.com/apache/tvm/blob/main/python/tvm/_ffi/_ctypes/ndarray.py>`__,
  `Cython <https://github.com/apache/tvm/blob/main/python/tvm/_ffi/_cython/ndarray.pxi>`__
* mpi4py: `Cython <https://github.com/mpi4py/mpi4py/blob/master/src/mpi4py/MPI/asdlpack.pxi>`_
