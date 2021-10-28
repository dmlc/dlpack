import ctypes

_c_str_dltensor = b"dltensor"


class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLCUDAManaged = 13
    kDLOneAPI = 14

    def __str__(self):
        return {
            self.kDLCPU : "CPU",
            self.kDLCUDA: "CUDA",
            self.kDLCUDAHost: "CUDAHost",
            self.kDLOpenCL: "OpenCL",
            self.kDLVulkan: "Vulkan",
            self.kDLMetal: "Metal",
            self.kDLVPI: "VPI",
            self.kDLROCM: "ROCM",
            self.kDLROCMHost: "ROMCHost",
            self.kDLCUDAManaged: "CUDAManaged",
            self.kDLOneAPI: "oneAPI",
            }[self.value]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", DLDeviceType),
        ("device_id", ctypes.c_int),
    ]


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
    kDLComplex = 5

    def __str__(self):
        return {
            self.kDLInt: "int",
            self.kDLUInt: "uint",
            self.kDLFloat: "float",
            self.kDLBfloat: "bfloat",
            self.kDLComplex: "complex"
        }[self.value]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]
    TYPE_MAP = {
        "bool": (DLDataTypeCode.kDLUInt, 1, 1),
        "int8": (DLDataTypeCode.kDLInt, 8, 1),
        "int16": (DLDataTypeCode.kDLInt, 16, 1),
        "int32": (DLDataTypeCode.kDLInt, 32, 1),
        "int64": (DLDataTypeCode.kDLInt, 64, 1),
        "uint8": (DLDataTypeCode.kDLUInt, 8, 1),
        "uint16": (DLDataTypeCode.kDLUInt, 16, 1),
        "uint32": (DLDataTypeCode.kDLUInt, 32, 1),
        "uint64": (DLDataTypeCode.kDLUInt, 64, 1),
        "float16": (DLDataTypeCode.kDLFloat, 16, 1),
        "float32": (DLDataTypeCode.kDLFloat, 32, 1),
        "float64": (DLDataTypeCode.kDLFloat, 64, 1),
        "complex64": (DLDataTypeCode.kDLComplex, 64, 1),
        "complex128": (DLDataTypeCode.kDLComplex, 128, 1)
    }


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

    @property
    def itemsize(self):
        return self.dtype.lanes * self.dtype.bits // 8

    @property
    def __array_interface__(self):
        shape = tuple(self.shape[dim] for dim in range(self.ndim))
        if self.strides:
            strides = tuple(
                self.strides[dim] * self.itemsize for dim in range(self.ndim)
            )
        else:
            # Array is compact, make it numpy compatible.
            strides = []
            for i, s in enumerate(shape):
                cumulative = 1
                for e in range(i + 1, self.ndim):
                    cumulative *= shape[e]
                strides.append(cumulative * self.itemsize)
            strides = tuple(strides)
        typestr = "|" + str(self.dtype.type_code)[0] + str(self.itemsize)
        return dict(
            version=3,
            shape=shape,
            strides=strides,
            data=(self.data, True),
            offset=self.byte_offset,
            typestr=typestr,
        )


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]

    @property
    def __array_interface__(self):
        return self.dl_tensor.__array_interface__