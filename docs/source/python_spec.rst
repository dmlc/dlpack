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
2. ``__dlpack__(self, stream=None, version=None)`` and
   ``__dlpack_device__(version=None)`` methods on the
   array object, which will be called from within ``from_dlpack``, to query
   what device the array is on (may be needed to pass in the correct
   stream, e.g. in the case of multiple GPUs) and to access the data.


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
``DLManagedTensor`` that is compatible with the DLPack and DLPack ABI
version requested by the consumer. It will be consumed immediately
within ``from_dlpack`` - therefore it is consumed exactly once, and it
will not be visible to users of the Python API.

The producer must set the ``PyCapsule`` name to ``"dltensor"`` if ABI
version 1 is requested and ``"versioned_dltensor"`` if ABI version >= 2
is requested so that it can be inspected by name, and set
``PyCapsule_Destructor`` that calls the ``deleter`` of the ``DLManagedTensor``
when the ``"dltensor"``-named or ``"versioned_dltensor"``-named capsule is
no longer needed.

The consumer must transfer ownership of the ``DLManangedTensor`` from the
capsule to its own object. It does so by renaming the capsule to
``"used_dltensor"`` to ensure that ``PyCapsule_Destructor`` will not get
called (ensured if ``PyCapsule_Destructor`` calls ``deleter`` only for
capsules whose name is ``"dltensor"`` or ``"versioned_dltensor"``), but
the ``deleter`` of the ``DLManagedTensor`` will be called by the destructor
of the consumer library object created to own the ``DLManagerTensor`` obtained
from the capsule. Below is an example of the capsule deleter written in the
Python C API which is called either when the refcount on the capsule named
``"dltensor"`` reaches zero or the consumer decides to deallocate its array:

.. code-block:: C

   static void dlpack_capsule_deleter(PyObject *self){
      if (PyCapsule_IsValid(self, "used_dltensor")) {
         return; /* Do nothing if the capsule has been consumed. */
      }

      /* an exception may be in-flight, we must save it in case we create another one */
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);

      /* Note that if ABI version >= 2 is used, the capsule will be named "versioned_dltensor" */
      DLManagedTensor *managed = (DLManagedTensor *)PyCapsule_GetPointer(self, "dltensor");
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

When the ``strides`` field in the ``DLTensor`` struct is ``NULL``, it indicates a
row-major compact array. If the array is of size zero, the data pointer in
``DLTensor`` should be set to either ``NULL`` or ``0``.

For further details on DLPack design and how to implement support for it,
refer to https://github.com/dmlc/dlpack. For details on ABI compatibility,
refer to :ref:`future-abi-compat`. To upgrade to the new ABI (version 2),
refer to :ref:`upgrade-policy`.

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

.. _upgrade-policy:

Upgrade Policy
~~~~~~~~~~~~~~

DLPack has been upgraded to ABI version 2. ABI version 1 did not contain any
version info in the ``DLTensor`` or ``DLManagedTensor`` structs. This has been added
now and can be used to check if the producer's ``DLManagedTensor`` is compatible with
the consumer's ``DLManagedTensor``. This section gives a path for Python libraries
to upgrade to the new ABI (while preserving support for the old ABI):

* ``__dlpack__`` should accept a ``version`` keyword (a Python tuple
  ``(dlpack_version, dlpack_abi_version)``) which is set to ``None`` by default.
  Consumers can use this kwarg to request certain versions of DLPack. If
  ``version=None`` or the ABI version 1 is requested:

  * a capsule named ``"dltensor"`` which uses the old ABI should be returned
    (if the producer wants to keep supporting it) or
  * a ``BufferError`` should be raised (if the producer doesn't want to keep
    support for the old ABI)

  Otherwise, a capsule named ``"versioned_dltensor"`` should be returned which
  uses the new ABI. If the requested version is not supported, ``__dlpack__``
  should raise a ``BufferError``.
* Consumers should pass a ``version`` keyword to the ``__dlpack__`` and
  ``__dlpack_device__`` methods requesting for a particular DLPack version and
  DLPack ABI version.
* If the ``__dlpack__`` method doesn't accept the ``version`` kwarg, the
  consumer should fallback to the old API i.e. passing no arguments to
  ``__dlpack__``. The consumers can check the capsule name: if a ``"dltensor"``
  capsule is received, the old ABI can be used to import data or if a
  ``"versioned_dltensor"`` is received, the version in the ``DLTensor`` struct
  can be used to check for compatibility.


.. _future-abi-compat:

Future ABI Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

DLPack now contains a ``DLPACK_VERSION`` and ``DLPACK_ABI_VERSION`` macro that defines
the current DLPack and ABI version respectively. Since DLPack ABI version 2,
``DLTensor`` contains a ``version`` field with ``dlpack`` and ``abi`` version fields
which can be used by the consumers to check for compatibility. In case of an ABI
break in the future, the consumers can request the ``__dlpack__`` method to
return a capsule compatible with a particular DLPack and ABI version by passing
a ``version`` keyword. ``version`` should be a tuple where the first element is
the requested DLPack version and the second element is the requested ABI version.
If the producer doesn't support the given versions, it should raise a
``BufferError``.

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
