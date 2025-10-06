/*!
 *  Copyright (c) 2017 -  by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

/**
 * \brief Compatibility with C++
 */
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current major version of dlpack */
#define DLPACK_MAJOR_VERSION 1

/*! \brief The current minor version of dlpack */
#define DLPACK_MINOR_VERSION 1

/*! \brief DLPACK_DLL prefix for windows */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief The DLPack version.
 *
 * A change in major version indicates that we have changed the
 * data layout of the ABI - DLManagedTensorVersioned.
 *
 * A change in minor version indicates that we have added new
 * code, such as a new device type, but the ABI is kept the same.
 *
 * If an obtained DLPack tensor has a major version that disagrees
 * with the version number specified in this header file
 * (i.e. major != DLPACK_MAJOR_VERSION), the consumer must call the deleter
 * (and it is safe to do so). It is not safe to access any other fields
 * as the memory layout will have changed.
 *
 * In the case of a minor version mismatch, the tensor can be safely used as
 * long as the consumer knows how to interpret all fields. Minor version
 * updates indicate the addition of enumeration values.
 */
typedef struct {
  /*! \brief DLPack major version. */
  uint32_t major;
  /*! \brief DLPack minor version. */
  uint32_t minor;
} DLPackVersion;

/*!
 * \brief The device type in DLDevice.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*!
   * \brief Pinned CUDA CPU memory by cudaMallocHost
   */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics. */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*!
   * \brief Pinned ROCm CPU memory allocated by hipMallocHost
   */
  kDLROCMHost = 11,
  /*!
   * \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
  /*!
   * \brief CUDA managed/unified memory allocated by cudaMallocManaged
   */
  kDLCUDAManaged = 13,
  /*!
   * \brief Unified shared memory allocated on a oneAPI non-partititioned
   * device. Call to oneAPI runtime is required to determine the device
   * type, the USM allocation type and the sycl context it is bound to.
   *
   */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard. */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
  /*! \brief Microsoft MAIA devices */
  kDLMAIA = 17,
  /*! \brief AWS Trainium */
  kDLTrn = 18,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*!
   * \brief The device index.
   * For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*! \brief signed integer */
  kDLInt = 0U,
  /*! \brief unsigned integer */
  kDLUInt = 1U,
  /*! \brief IEEE floating point */
  kDLFloat = 2U,
  /*!
   * \brief Opaque handle type, reserved for testing purposes.
   * Frameworks need to agree on the handle data type for the exchange to be well-defined.
   */
  kDLOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kDLBfloat = 4U,
  /*!
   * \brief complex number
   * (C/C++/Python layout: compact struct per complex number)
   */
  kDLComplex = 5U,
  /*! \brief boolean */
  kDLBool = 6U,
  /*! \brief FP8 data types */
  kDLFloat8_e3m4 = 7U,
  kDLFloat8_e4m3 = 8U,
  kDLFloat8_e4m3b11fnuz = 9U,
  kDLFloat8_e4m3fn = 10U,
  kDLFloat8_e4m3fnuz = 11U,
  kDLFloat8_e5m2 = 12U,
  kDLFloat8_e5m2fnuz = 13U,
  kDLFloat8_e8m0fnu = 14U,
  /*! \brief FP6 data types
   * Setting bits != 6 is currently unspecified, and the producer must ensure it is set
   * while the consumer must stop importing if the value is unexpected.
   */
  kDLFloat6_e2m3fn = 15U,
  kDLFloat6_e3m2fn = 16U,
  /*! \brief FP4 data types
   * Setting bits != 4 is currently unspecified, and the producer must ensure it is set
   * while the consumer must stop importing if the value is unexpected.
   */
  kDLFloat4_e2m1fn = 17U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 *   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
 *   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
 *   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
 *
 *  When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
 *  for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
 */
typedef struct {
  /*!
   * \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
  uint8_t code;
  /*!
   * \brief Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /*! \brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*!
   * \brief The data pointer points to the allocated data. This will be CUDA
   * device pointer or cl_mem handle in OpenCL. It may be opaque on some device
   * types. This pointer is always aligned to 256 bytes as in CUDA. The
   * `byte_offset` field should be used to point to the beginning of the data.
   *
   * Note that as of Nov 2021, multiple libraries (CuPy, PyTorch, TensorFlow,
   * TVM, perhaps others) do not adhere to this 256 byte alignment requirement
   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
   * (after which this note will be updated); at the moment it is recommended
   * to not rely on the data pointer being correctly aligned.
   *
   * For given DLTensor, the size of memory required to store the contents of
   * data is calculated as follows:
   *
   * \code{.c}
   * static inline size_t GetDataSize(const DLTensor* t) {
   *   size_t size = 1;
   *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * \endcode
   *
   * Note that if the tensor is of size zero, then the data pointer should be
   * set to `NULL`.
   */
  void* data;
  /*! \brief The device of the tensor */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*!
   * \brief strides of the tensor (in number of elements, not bytes)
   *  can be NULL, indicating tensor is compact and row-majored.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, manage memory of DLTensor. This data structure is
 *  intended to facilitate the borrowing of DLTensor by another framework. It is
 *  not meant to transfer the tensor. When the borrowing framework doesn't need
 *  the tensor, it should call the deleter to notify the host that the resource
 *  is no longer needed.
 *
 * \note This data structure is used as Legacy DLManagedTensor
 *       in DLPack exchange and is deprecated after DLPack v0.8
 *       Use DLManagedTensorVersioned instead.
 *       This data structure may get renamed or deleted in future versions.
 *
 * \sa DLManagedTensorVersioned
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
  /*! \brief the context of the original host framework of DLManagedTensor in
   *   which DLManagedTensor is used in the framework. It can also be NULL.
   */
  void * manager_ctx;
  /*!
   * \brief Destructor - this should be called
   * to destruct the manager_ctx  which backs the DLManagedTensor. It can be
   * NULL if there is no way for the caller to provide a reasonable destructor.
   * The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensor * self);
} DLManagedTensor;

// bit masks used in the DLManagedTensorVersioned

/*! \brief bit mask to indicate that the tensor is read only. */
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)

/*!
 * \brief bit mask to indicate that the tensor is a copy made by the producer.
 *
 * If set, the tensor is considered solely owned throughout its lifetime by the
 * consumer, until the producer-provided deleter is invoked.
 */
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

/*!
 * \brief bit mask to indicate that whether a sub-byte type is packed or padded.
 *
 * The default for sub-byte types (ex: fp4/fp6) is assumed packed. This flag can
 * be set by the producer to signal that a tensor of sub-byte type is padded.
 */
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2UL)

/*!
 * \brief A versioned and managed C Tensor object, manage memory of DLTensor.
 *
 * This data structure is intended to facilitate the borrowing of DLTensor by
 * another framework. It is not meant to transfer the tensor. When the borrowing
 * framework doesn't need the tensor, it should call the deleter to notify the
 * host that the resource is no longer needed.
 *
 * \note This is the current standard DLPack exchange data structure.
 */
typedef struct DLManagedTensorVersioned {
  /*!
   * \brief The API and ABI version of the current managed Tensor
   */
  DLPackVersion version;
  /*!
   * \brief the context of the original host framework.
   *
   * Stores DLManagedTensorVersioned is used in the
   * framework. It can also be NULL.
   */
  void *manager_ctx;
  /*!
   * \brief Destructor.
   *
   * This should be called to destruct manager_ctx which holds the DLManagedTensorVersioned.
   * It can be NULL if there is no way for the caller to provide a reasonable
   * destructor. The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensorVersioned *self);
  /*!
   * \brief Additional bitmask flags information about the tensor.
   *
   * By default the flags should be set to 0.
   *
   * \note Future ABI changes should keep everything until this field
   *       stable, to ensure that deleter can be correctly called.
   *
   * \sa DLPACK_FLAG_BITMASK_READ_ONLY
   * \sa DLPACK_FLAG_BITMASK_IS_COPIED
   */
  uint64_t flags;
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

//--------------------------------------------------------------------
// DLPack C functions for speed exchange
//--------------------------------------------------------------------
/*!
 * \brief A generic C-style allocator that exposes allocation of a Tensor/Array.
 *
 * This information can then be used to set allocators of a callee to run allocations.
 * This information can then be used to set the callee's allocator to perform allocations.
 * This function can be exposed by the framework through the DLPackExchangeAPI.
 *
 * This particular function does not assume a Python environment; as a result,
 * the error handling mechanism is different from Python-related functions.
 *
 * \param prototype The prototype DLTensor to offer details about the device and shape.
 *                  Other field information will be ignored during allocation.
 * \param out The output DLManagedTensorVersioned.
 * \param error_ctx The context to set the error.
 * \param SetError The function to set the error.
 * \return 0 on success, -1 on failure.
 *         The callee should call SetError(error_ctx, kind, message) to set the error kind and message.
 * \note Error propagation via SetError.
 *
 * \sa DLPackExchangeAPI
 */
typedef int (*DLPackManagedTensorAllocator)(                                //
  DLTensor* prototype, DLManagedTensorVersioned** out, void* error_ctx,     //
  void (*SetError)(void* error_ctx, const char* kind, const char* message)  //
);

/*!
 * \brief Exports a PyObject* Tensor/NDArray to a DLManagedTensorVersioned.
 *
 * This function is a C-style function pointer to quickly convert a PyObject* Tensor/NDArray
 * to a DLManagedTensorVersioned without going through the Python interpreter.
 *
 * This function does not perform any stream synchronization. The consumer should query
 * DLPackCurrentWorkStream to get the current work stream and launch kernels on it.
 *
 * This function is exposed by the framework through the DLPackExchangeAPI.
 *
 * This information can then be picked up by importers and libraries to perform a fast conversion.
 * This function should not throw any exceptions; if it fails, it should return -1 and
 * set the error message via PyErr_SetXXX.
 *
 * \param py_object The Python object to convert; this should be PyObject*.
 *                  We use void* to avoid dependency on Python.h.
 *
 * \param out The output DLManagedTensorVersioned.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 * \note We use void* to avoid dependency on Python.h, so this specific type is
 *       not dependent on Python.h and can be copied to dlpack.h.
 *
 * \sa DLPackExchangeAPI, DLPackCurrentWorkStream
 */
typedef int (*DLPackManagedTensorFromPyObjectNoSync)(                 //
  void* py_object,                                                    //
  DLManagedTensorVersioned** out                                      //
);

/*!
 * \brief Exports a PyObject* Tensor/NDArray to a DLTensor whose space is pre-allocated on stack.
 *
 * This function is a C-style function pointer to quickly convert a PyObject* Tensor/NDArray
 * to a DLTensor whose space is pre-allocated on stack without going through the Python interpreter.
 *
 * This is an non-owning conversion, the producer still owns the memory of data, strides, shape.
 * The liveness of DLTensor is only guaranteed until the consumer returns control to the caller.
 *
 * In the context of this function, we expect the producer to allocated space for data, strides and shape.
 *
 * This function does not perform any stream synchronization. The consumer should query
 * DLPackCurrentWorkStream to get the current work stream and launch kernels on it.
 *
 * This function is useful when the consumer do not need to retain the tensor memory.
 * It generally can provide about 2x faster conversion than DLPackManagedTensorFromPyObjectNoSync.
 *
 * For cases where consumer may needs to reorganize the tensor memory via temporary managed copy,
 * DLPackManagedTensorFromPyObjectNoSync should be used.
 *
 * This function is exposed by the framework through the DLPackExchangeAPI.
 *
 * This information can then be picked up by importers and libraries to perform a fast conversion.
 * This function should not throw any exceptions; if it fails, it should return -1 and
 * set the error message via PyErr_SetXXX.
 *
 * \param py_object The Python object to convert; this should be PyObject*.
 *                  We use void* to avoid dependency on Python.h.
 *
 * \param out The output DLTensor, whose space is pre-allocated on stack.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 * \note We use void* to avoid dependency on Python.h, so this specific type is
 *       not dependent on Python.h and can be copied to dlpack.h.
 *
 * \sa DLPackExchangeAPI, DLPackCurrentWorkStream
 */
typedef int (*DLPackDLTensorFromPyObjectNoSync)(                      //
  void* py_object,                                                    //
  DLTensor* out                                                       //
);

/*!
 * \brief Obtain the current work stream of a device.
 *
 * This function is a C-style function pointer to obtain the current work stream
 * of a device for frameworks that rely on a context manager to manage the stream.
 * For example, it should map to torch.cuda.current_stream in PyTorch.
 *
 * This function can be set to NULL if the framework does not rely on a context manager
 * to manage the stream. However, we encourage frameworks to provide this function
 * if possible.
 *
 * As if this field is not set, likely consumer cannot safely do stream based
 * exchange based on the
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param out_current_stream The output current work stream.
 * \return 0 on success, -1 on failure.
 *
 * \sa DLPackExchangeAPI
 */
typedef int (*DLPackCurrentWorkStream)(                         //
  DLDeviceType device_type,                                     //
  int32_t device_id,                                            //
  void** out_current_stream                                     //
);

/*!
 * \brief Imports a DLManagedTensorVersioned to a PyObject* Tensor/NDArray.
 *
 * This function is a C-style function pointer to quickly convert a DLManagedTensorVersioned
 * to a PyObject* without going through the Python Interpreter.
 *
 * This function does not perform any stream synchronization.
 *
 * This function is exposed by the framework through the DLPackExchangeAPI.
 *
 * \param tensor The DLManagedTensorVersioned to convert.
 * \param out_py_object The output Python object.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 * \note We use void* to avoid dependency on Python.h, so this specific type is
 *       not dependent on Python.h and can be copied to dlpack.h.
 *
 * \sa DLPackExchangeAPI
 */
typedef int (*DLPackManagedTensorToPyObjectNoSync)(                     //
  DLManagedTensorVersioned* tensor, void** out_py_object                //
);

/*!
 * \brief Framework-specific function pointers table for DLPack exchange.
 *
 * Guidelines for leveraging DLPackExchangeAPI:
 *
 * There are generally two kinds of consumer needs for DLPack exchange:
 * - N0: library support, where consumer.kernel(x, y, z) would like to run a kernel
 *       with the data from x, y, z. The consumer is also expected to run the kernel with the same
 *       stream context as the producer. For example, when x, y, z is torch.Tensor,
 *       consumer should query exchange_api->current_work_stream to get the
 *       current stream and launch the kernel with the same stream.
 *       This setup is necessary for no synchronization in kernel launch and maximum compatibility
 *       with CUDA graph capture in the producer.
 *       This is the desirable behavior for library extension support for frameworks like PyTorch.
 * - N1: data ingestion and retention
 *
 * Note that obj.__dlpack__() API should provide useful ways for N1.
 * The primary focus of the current DLPackExchangeAPI is to enable faster exchange N0
 * with the support of the function pointer current_work_stream.
 *
 * Array/Tensor libraries should statically create and initialize this structure
 * then return a pointer to DLPackExchangeAPI as an int value in Tensor/Array.
 * The DLPackExchangeAPI* should stay alive throughout the lifetime of the process.
 *
 * One simple way to do so is to create a static instance of DLPackExchangeAPI
 * within the framework and return a pointer to it. The following code
 * shows an example to do so in C++. It should also be reasonably easy
 * to do so in other languages.
 *
 * \code
 * struct MyDLPackExchangeAPI : public DLPackExchangeAPI {
 *   MyDLPackExchangeAPI() {
 *     version.major = DLPACK_MAJOR_VERSION;
 *     version.minor = DLPACK_MINOR_VERSION;
 *     managed_tensor_allocator = MyDLPackManagedTensorAllocator;
 *     managed_tensor_from_py_object_no_sync = MyDLPackManagedTensorFromPyObjectNoSync;
 *     managed_tensor_to_py_object_no_sync = MyDLPackManagedTensorToPyObjectNoSync;
 *     dltensor_from_py_object_no_sync = MyDLPackDLTensorFromPyObjectNoSync;
 *     current_work_stream = MyDLPackCurrentWorkStream;
 *     prev_version_api = nullptr;
 *  }
 *
 *  static const DLPackExchangeAPI* Global() {
 *     static MyDLPackExchangeAPI inst;
 *     return &inst;
 *  }
 * };
 * \endcode
 *
 * Each framework should attach a dunder `__c_dlpack_exchange_api__` integer
 * to point to the DLPackExchangeAPI* pointer.
 *
 * Importantly, the attribute should be attached to the class of the Tensor, not the instance.
 *
 * mypackage.Tensor.__c_dlpack_exchange_api__ = MyPackageDLPackExchangeAPI
 *
 * or equivalently:
 *
 * type(tensor_obj).__c_dlpack_exchange_api__ = MyPackageDLPackExchangeAPI
 */
struct DLPackExchangeAPI {
  /*!
   * \brief The current DLPack version.
   */
  DLPackVersion version;
  /*!
   * \brief Optional pointer to an older DLPackExchangeAPI in the chain.
   *
   * It should be set to NULL if the framework does not support older versions.
   *
   * \sa DLPackExchangeAPI
   */
  struct DLPackExchangeAPI* prev_version_api;
  /*!
   * \brief Framework-specific function pointer for DLPackManagedTensorAllocator
   * \sa DLPackManagedTensorAllocator
   */
  DLPackManagedTensorAllocator managed_tensor_allocator;
  /*!
   * \brief Framework-specific function pointer for DLPackManagedTensorFromPyObject
   * \sa DLPackManagedTensorFromPyObject
   */
  DLPackManagedTensorFromPyObjectNoSync managed_tensor_from_py_object_no_sync;
  /*!
   * \brief Framework-specific function pointer for DLPackManagedTensorToPyObject
   * \sa DLPackManagedTensorToPyObject
   */
  DLPackManagedTensorToPyObjectNoSync managed_tensor_to_py_object_no_sync;
  /*!
   * \brief Framework-specific function pointer for DLPackDLTensorFromPyObject
   * \sa DLPackDLTensorFromPyObjectNoSync
   */
  DLPackDLTensorFromPyObjectNoSync dltensor_from_py_object_no_sync;
  /*!
   * \brief Framework-specific function pointer for DLPackCurrentWorkStream
   *
   * This function can be set to NULL if the framework does not rely on context manager to manage the stream.
   *
   * \sa DLPackCurrentWorkStream
   */
  DLPackCurrentWorkStream current_work_stream;
};

#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_DLPACK_H_
