.. _release-notes:

DLPack Change Log
=================

This file records the changes in DLPack in reverse chronological order.

v0.7
~~~~

Note: This release contains ABI breaking changes.

- ABI version has been added as ``DLPACK_ABI_VERSION`` macro in the header file.
- Support for OneAPI (``kDLOneAPI``), WebGPU (``kDLWebGPU``), and Hexagon
  (``kDLHexagon``) devices has been added.
- Two new structs with a field to export the version info and the readonly
  flag have been added: ``DLTensorVersioned`` and ``DLManagedTensorVersioned``.
  New implementations should use these structs over ``DLTensor`` and
  ``DLManagedTensor``. If you have already added support for DLPack, it should
  be updated to use the new structs instead (warning: this is an ABI breaking
  change for C/C++ libraries. Python libraries should follow the
  :ref:`future-abi-compat` section to upgrade to the new ABI without breaking
  backward-compatibility).

v0.6
~~~~

- Support for ROCm host memory and CUDA managed memory has been added.

v0.5
~~~~

- Devices kDLGPU and kDLCPUPinned have been renamed to kDLCUDA and kDLCUDAHost
  respectively.

v0.4
~~~~

- OpaqueHandle type
- Complex support
- Rename DLContext -> DLDevice
  - This requires dependent frameworks to upgrade the type name.
  - The ABI is backward compatible, as it is only change of constant name.

v0.3
~~~~

- Add bfloat16
- Vulkan support


v0.2
~~~~

- New device types
  - kDLMetal for Apple Metal device
  - kDLVPI for verilog simulator memory
  - kDLROCM for AMD GPUs
- Add prefix DL to all enum constant values
  - This requires dependent frameworks to upgrade their reference to these constant
  - The ABI is compatible, as it is only change of constant name.
- Add DLManagedTensor structure for borrowing tensors

v0.1
~~~~

- Finalize DLTensor structure
