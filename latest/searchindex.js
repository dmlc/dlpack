Search.setIndex({"docnames": ["c_api", "index", "python_spec"], "filenames": ["c_api.rst", "index.rst", "python_spec.rst"], "titles": ["C API (<code class=\"docutils literal notranslate\"><span class=\"pre\">dlpack.h</span></code>)", "Welcome to DLPack\u2019s documentation!", "Python Specification for DLPack"], "terms": {"dlpack_extern_c": 0, "compat": [0, 1], "dlpack_major_vers": 0, "The": [0, 1, 2], "current": 0, "major": [0, 1, 2], "version": [0, 1, 2], "dlpack_minor_vers": 0, "minor": 0, "dlpack_dll": 0, "prefix": 0, "window": 0, "dlpack_flag_bitmask_read_onli": 0, "bit": [0, 2], "mask": 0, "indic": [0, 2], "tensor": [0, 2], "i": [0, 1, 2], "read": [0, 1], "onli": [0, 1, 2], "dlpack_flag_bitmask_is_copi": [0, 2], "copi": [0, 2], "made": [0, 2], "produc": [0, 2], "If": [0, 2], "set": [0, 2], "consid": [0, 1, 2], "sole": 0, "own": [0, 2], "throughout": 0, "its": [0, 2], "lifetim": 0, "consum": [0, 2], "until": 0, "provid": [0, 2], "delet": [0, 2], "invok": 0, "enum": [0, 2], "dldevicetyp": 0, "devic": [0, 1, 2], "type": [0, 1, 2], "dldevic": 0, "valu": [0, 2], "kdlcpu": [0, 2], "cpu": [0, 1], "kdlcuda": 0, "cuda": [0, 1, 2], "gpu": [0, 2], "kdlcudahost": 0, "pin": 0, "memori": [0, 1, 2], "cudamallochost": 0, "kdlopencl": 0, "opencl": [0, 1], "kdlvulkan": 0, "vulkan": [0, 1], "buffer": 0, "next": 0, "gener": 0, "graphic": 0, "kdlmetal": 0, "metal": [0, 1], "appl": 0, "kdlvpi": 0, "verilog": 0, "simul": 0, "kdlrocm": 0, "rocm": [0, 1, 2], "amd": 0, "kdlrocmhost": 0, "alloc": [0, 1, 2], "hipmallochost": 0, "kdlextdev": 0, "reserv": 0, "extens": 0, "us": [0, 1, 2], "quickli": 0, "test": 0, "semant": [0, 1], "can": [0, 1, 2], "differ": 0, "depend": [0, 2], "implement": [0, 1], "kdlcudamanag": 0, "manag": [0, 2], "unifi": 0, "cudamallocmanag": 0, "kdloneapi": 0, "share": [0, 2], "oneapi": 0, "non": [0, 2], "partitit": 0, "call": [0, 2], "runtim": [0, 2], "requir": [0, 1, 2], "determin": 0, "usm": 0, "sycl": 0, "context": 0, "bound": 0, "kdlwebgpu": 0, "support": [0, 1, 2], "webgpu": [0, 1], "standard": [0, 1, 2], "kdlhexagon": 0, "qualcomm": 0, "hexagon": [0, 1], "dsp": 0, "kdlmaia": 0, "microsoft": 0, "maia": 0, "dldatatypecod": 0, "code": [0, 2], "option": 0, "dldatatyp": 0, "kdlint": 0, "sign": 0, "integ": [0, 2], "kdluint": 0, "unsign": 0, "kdlfloat": 0, "ieee": 0, "float": 0, "point": 0, "kdlopaquehandl": 0, "opaqu": [0, 1], "handl": [0, 2], "purpos": 0, "framework": [0, 1], "need": [0, 1, 2], "agre": 0, "data": [0, 1], "exchang": [0, 1], "well": [0, 2], "defin": [0, 2], "kdlbfloat": 0, "bfloat16": 0, "kdlcomplex": 0, "complex": [0, 1], "number": [0, 2], "python": [0, 1], "layout": [0, 1, 2], "compact": [0, 2], "per": 0, "kdlbool": 0, "boolean": [0, 1], "dlpackvers": 0, "A": [0, 2], "chang": 0, "we": 0, "have": [0, 2], "abi": [0, 1], "dlmanagedtensorvers": [0, 2], "ad": [0, 1, 2], "new": [0, 1, 2], "kept": [0, 2], "same": [0, 2], "an": [0, 1, 2], "obtain": [0, 2], "ha": [0, 2], "disagre": 0, "specifi": 0, "thi": [0, 2], "header": [0, 1], "file": 0, "e": [0, 1, 2], "must": [0, 2], "safe": [0, 2], "do": [0, 2], "so": [0, 2], "It": [0, 1, 2], "access": [0, 1, 2], "ani": [0, 1, 2], "other": [0, 1, 2], "field": [0, 1, 2], "In": [0, 1, 2], "case": [0, 1, 2], "mismatch": 0, "long": 0, "know": 0, "how": [0, 2], "interpret": 0, "all": 0, "updat": 0, "addit": 0, "public": 0, "member": [0, 2], "uint32_t": 0, "oper": 0, "device_typ": 0, "int32_t": 0, "device_id": [0, 2], "index": [0, 1], "For": [0, 2], "vanilla": 0, "0": [0, 2], "hold": 0, "assum": [0, 1, 2], "follow": [0, 2], "nativ": 0, "endian": 0, "ness": 0, "explicit": [0, 2], "error": 0, "messag": 0, "should": [0, 1, 2], "rais": [0, 2], "when": [0, 2], "attempt": 0, "export": 0, "arrai": [0, 1, 2], "exampl": [0, 2], "type_cod": 0, "2": 0, "32": 0, "lane": 0, "1": [0, 1, 2], "float4": 0, "vector": 0, "4": 0, "int8": 0, "8": 0, "std": 0, "5": 0, "64": 0, "bool": 0, "6": 0, "common": [0, 2], "librari": [0, 2], "convent": 0, "underli": 0, "storag": 0, "size": [0, 2], "uint8_t": 0, "base": 0, "keep": [0, 2], "instead": 0, "minim": [0, 1], "footprint": 0, "one": [0, 1, 2], "choic": 0, "ar": [0, 2], "16": 0, "uint16_t": 0, "dltensor": [0, 2], "plain": 0, "object": [0, 2], "doe": [0, 1, 2], "void": [0, 2], "pointer": [0, 2], "cl_mem": 0, "mai": [0, 2], "some": [0, 1, 2], "alwai": 0, "align": [0, 1], "256": 0, "byte": 0, "byte_offset": 0, "begin": 0, "note": [0, 2], "nov": 0, "2021": 0, "multipli": 0, "cupi": [0, 1, 2], "pytorch": [0, 1, 2], "tensorflow": [0, 1, 2], "tvm": [0, 1, 2], "perhap": [0, 2], "adher": 0, "alig": 0, "fix": 0, "after": [0, 2], "which": [0, 2], "moment": 0, "recommend": [0, 2], "reli": 0, "being": [0, 2], "correctli": 0, "given": 0, "store": 0, "content": 0, "calcul": 0, "static": [0, 2], "inlin": 0, "size_t": 0, "getdatas": 0, "const": 0, "t": [0, 2], "tvm_index_t": 0, "ndim": 0, "shape": [0, 2], "dtype": 0, "7": 0, "return": [0, 1, 2], "zero": [0, 2], "null": [0, 2], "dimens": 0, "int64_t": 0, "stride": [0, 1, 2], "element": 0, "row": [0, 1, 2], "uint64_t": 0, "offset": 0, "dlmanagedtensor": [0, 2], "structur": [0, 1], "intend": 0, "facilit": 0, "borrow": 0, "anoth": 0, "meant": 0, "transfer": 0, "doesn": 0, "notifi": 0, "host": 0, "resourc": 0, "longer": [0, 2], "legaci": [0, 1], "deprec": [0, 2], "v0": 0, "get": [0, 2], "renam": [0, 2], "futur": 0, "dl_tensor": 0, "manager_ctx": [0, 2], "origin": [0, 2], "also": [0, 1, 2], "self": [0, 2], "destructor": [0, 2], "destruct": 0, "back": 0, "wai": [0, 2], "caller": [0, 2], "reason": [0, 2], "argument": [0, 2], "flag": [0, 1, 2], "bitmask": 0, "inform": 0, "about": [0, 2], "By": 0, "default": [0, 2], "everyth": [0, 1], "stabl": [0, 1], "ensur": [0, 2], "order": [1, 2], "ndarrai": 1, "system": 1, "interact": 1, "varieti": 1, "allow": 1, "between": [1, 2], "develop": 1, "input": [1, 2], "from": [1, 2], "mani": 1, "deep": 1, "learn": 1, "core": 1, "highlight": 1, "includ": 1, "minimum": 1, "simpl": 1, "design": [1, 2], "cross": [1, 2], "hardwar": [1, 2], "vpi": 1, "alreadi": 1, "wide": 1, "commun": 1, "adopt": [1, 2], "numpi": [1, 2], "mxnet": [1, 2], "mpi4pi": [1, 2], "paddl": [1, 2], "clean": 1, "c": [1, 2], "mean": [1, 2], "you": 1, "creat": [1, 2], "languag": 1, "essenti": 1, "build": 1, "jit": 1, "aot": 1, "compil": 1, "main": 1, "rational": 1, "drop": 1, "consider": 1, "api": [1, 2], "focu": 1, "while": [1, 2], "still": 1, "g": [1, 2], "platform": 1, "normal": 1, "address": 1, "simplifi": 1, "remov": 1, "issu": 1, "avoid": [1, 2], "more": [1, 2], "could": 1, "expos": 1, "attribut": [1, 2], "__dlpack_info__": 1, "see": [1, 2], "34": 1, "72": 1, "clarifi": 1, "293": 1, "20338": 1, "comment": 1, "75": 1, "break": 1, "make": [1, 2], "hard": 1, "spec": [1, 2], "import": 1, "treat": [1, 2], "consortium": 1, "feedback": 1, "191": 1, "interfac": 1, "stream": [1, 2], "74": 1, "65": 1, "h": 1, "macro": 1, "enumer": 1, "struct": [1, 2], "specif": 1, "syntax": 1, "interchang": 1, "refer": 1, "search": 1, "page": [1, 2], "part": 2, "detail": 2, "found": 2, "under": 2, "mechan": 2, "offer": 2, "from_dlpack": 2, "function": 2, "accept": 2, "two": 2, "method": 2, "below": 2, "them": 2, "construct": 2, "contain": 2, "__dlpack__": 2, "__dlpack_device__": 2, "within": 2, "queri": 2, "what": 2, "pass": 2, "correct": 2, "multipl": 2, "describ": 2, "dens": 2, "n": 2, "dimension": 2, "user": 2, "y": 2, "x": 2, "possibl": 2, "view": 2, "both": 2, "therefor": 2, "mutat": 2, "mind": 2, "behaviour": 2, "mutabl": 2, "continu": 2, "just": 2, "like": 2, "via": 2, "protocol": 2, "live": 2, "request": 2, "buffererror": 2, "unless": 2, "through": 2, "keyword": 2, "appli": 2, "concept": 2, "howev": 2, "those": 2, "haven": 2, "been": 2, "synchron": 2, "wait": 2, "necessari": 2, "unnecessari": 2, "asynchron": 2, "execut": 2, "enabl": 2, "start": 2, "v2023": 2, "explicitli": 2, "disabl": 2, "though": 2, "mandat": 2, "larg": 2, "tri": 2, "discuss": 2, "here": 2, "becaus": 2, "coordin": 2, "diagram": 2, "dark": 2, "blue": 2, "light": 2, "grai": 2, "text": 2, "max_vers": 2, "signal": 2, "maxim": 2, "exist": 2, "try": 2, "dure": 2, "transit": 2, "period": 2, "rest": 2, "document": 2, "synonym": 2, "proper": 2, "done": 2, "choos": 2, "right": 2, "As": 2, "far": 2, "capsul": 2, "name": 2, "concern": 2, "used_dltensor": 2, "_version": 2, "suffix": 2, "pycapsul": 2, "immedi": 2, "exactli": 2, "onc": 2, "visibl": 2, "inspect": 2, "pycapsule_destructor": 2, "transer": 2, "ownership": 2, "whose": 2, "written": 2, "either": 2, "refcount": 2, "reach": 2, "decid": 2, "dealloc": 2, "dlpack_capsule_delet": 2, "pyobject": 2, "pycapsule_isvalid": 2, "noth": 2, "pycapsule_getpoint": 2, "pyerr_writeunrais": 2, "sai": 2, "beyond": 2, "boundari": 2, "gil": 2, "acquir": 2, "usual": 2, "py_decref": 2, "owner": 2, "free": 2, "arbitrari": 2, "array_dlpack_delet": 2, "leak": 2, "avail": 2, "happen": 2, "destroi": 2, "late": 2, "final": 2, "wa": 2, "indirectli": 2, "aliv": 2, "variabl": 2, "py_isiniti": 2, "pygilstate_st": 2, "state": 2, "pygilstate_ensur": 2, "": 2, "pymem_fre": 2, "py_xdecref": 2, "pygilstate_releas": 2, "further": 2, "github": 2, "com": 2, "dmlc": 2, "id": 2, "practic": 2, "guarante": 2, "look": 2, "up": 2, "actual": 2, "whether": 2, "certain": 2, "sever": 2, "cython": 2, "ctype": 2, "cffi": 2, "etc": 2, "wrapper": 2, "xla": 2}, "objects": {"": [[0, 0, 1, "c.DLDataType", "DLDataType"], [0, 2, 1, "c.DLDataTypeCode", "DLDataTypeCode"], [0, 0, 1, "c.DLDevice", "DLDevice"], [0, 2, 1, "c.DLDeviceType", "DLDeviceType"], [0, 0, 1, "c.DLManagedTensor", "DLManagedTensor"], [0, 0, 1, "c.DLManagedTensorVersioned", "DLManagedTensorVersioned"], [0, 4, 1, "c.DLPACK_DLL", "DLPACK_DLL"], [0, 4, 1, "c.DLPACK_EXTERN_C", "DLPACK_EXTERN_C"], [0, 4, 1, "c.DLPACK_FLAG_BITMASK_IS_COPIED", "DLPACK_FLAG_BITMASK_IS_COPIED"], [0, 4, 1, "c.DLPACK_FLAG_BITMASK_READ_ONLY", "DLPACK_FLAG_BITMASK_READ_ONLY"], [0, 4, 1, "c.DLPACK_MAJOR_VERSION", "DLPACK_MAJOR_VERSION"], [0, 4, 1, "c.DLPACK_MINOR_VERSION", "DLPACK_MINOR_VERSION"], [0, 0, 1, "c.DLPackVersion", "DLPackVersion"], [0, 0, 1, "c.DLTensor", "DLTensor"], [0, 3, 1, "c.DLDataTypeCode.kDLBfloat", "kDLBfloat"], [0, 3, 1, "c.DLDataTypeCode.kDLBool", "kDLBool"], [0, 3, 1, "c.DLDeviceType.kDLCPU", "kDLCPU"], [0, 3, 1, "c.DLDeviceType.kDLCUDA", "kDLCUDA"], [0, 3, 1, "c.DLDeviceType.kDLCUDAHost", "kDLCUDAHost"], [0, 3, 1, "c.DLDeviceType.kDLCUDAManaged", "kDLCUDAManaged"], [0, 3, 1, "c.DLDataTypeCode.kDLComplex", "kDLComplex"], [0, 3, 1, "c.DLDeviceType.kDLExtDev", "kDLExtDev"], [0, 3, 1, "c.DLDataTypeCode.kDLFloat", "kDLFloat"], [0, 3, 1, "c.DLDeviceType.kDLHexagon", "kDLHexagon"], [0, 3, 1, "c.DLDataTypeCode.kDLInt", "kDLInt"], [0, 3, 1, "c.DLDeviceType.kDLMAIA", "kDLMAIA"], [0, 3, 1, "c.DLDeviceType.kDLMetal", "kDLMetal"], [0, 3, 1, "c.DLDeviceType.kDLOneAPI", "kDLOneAPI"], [0, 3, 1, "c.DLDataTypeCode.kDLOpaqueHandle", "kDLOpaqueHandle"], [0, 3, 1, "c.DLDeviceType.kDLOpenCL", "kDLOpenCL"], [0, 3, 1, "c.DLDeviceType.kDLROCM", "kDLROCM"], [0, 3, 1, "c.DLDeviceType.kDLROCMHost", "kDLROCMHost"], [0, 3, 1, "c.DLDataTypeCode.kDLUInt", "kDLUInt"], [0, 3, 1, "c.DLDeviceType.kDLVPI", "kDLVPI"], [0, 3, 1, "c.DLDeviceType.kDLVulkan", "kDLVulkan"], [0, 3, 1, "c.DLDeviceType.kDLWebGPU", "kDLWebGPU"]], "DLDataType": [[0, 1, 1, "c.DLDataType.bits", "bits"], [0, 1, 1, "c.DLDataType.code", "code"], [0, 1, 1, "c.DLDataType.lanes", "lanes"]], "DLDataTypeCode": [[0, 3, 1, "c.DLDataTypeCode.kDLBfloat", "kDLBfloat"], [0, 3, 1, "c.DLDataTypeCode.kDLBool", "kDLBool"], [0, 3, 1, "c.DLDataTypeCode.kDLComplex", "kDLComplex"], [0, 3, 1, "c.DLDataTypeCode.kDLFloat", "kDLFloat"], [0, 3, 1, "c.DLDataTypeCode.kDLInt", "kDLInt"], [0, 3, 1, "c.DLDataTypeCode.kDLOpaqueHandle", "kDLOpaqueHandle"], [0, 3, 1, "c.DLDataTypeCode.kDLUInt", "kDLUInt"]], "DLDevice": [[0, 1, 1, "c.DLDevice.device_id", "device_id"], [0, 1, 1, "c.DLDevice.device_type", "device_type"]], "DLDeviceType": [[0, 3, 1, "c.DLDeviceType.kDLCPU", "kDLCPU"], [0, 3, 1, "c.DLDeviceType.kDLCUDA", "kDLCUDA"], [0, 3, 1, "c.DLDeviceType.kDLCUDAHost", "kDLCUDAHost"], [0, 3, 1, "c.DLDeviceType.kDLCUDAManaged", "kDLCUDAManaged"], [0, 3, 1, "c.DLDeviceType.kDLExtDev", "kDLExtDev"], [0, 3, 1, "c.DLDeviceType.kDLHexagon", "kDLHexagon"], [0, 3, 1, "c.DLDeviceType.kDLMAIA", "kDLMAIA"], [0, 3, 1, "c.DLDeviceType.kDLMetal", "kDLMetal"], [0, 3, 1, "c.DLDeviceType.kDLOneAPI", "kDLOneAPI"], [0, 3, 1, "c.DLDeviceType.kDLOpenCL", "kDLOpenCL"], [0, 3, 1, "c.DLDeviceType.kDLROCM", "kDLROCM"], [0, 3, 1, "c.DLDeviceType.kDLROCMHost", "kDLROCMHost"], [0, 3, 1, "c.DLDeviceType.kDLVPI", "kDLVPI"], [0, 3, 1, "c.DLDeviceType.kDLVulkan", "kDLVulkan"], [0, 3, 1, "c.DLDeviceType.kDLWebGPU", "kDLWebGPU"]], "DLManagedTensor": [[0, 1, 1, "c.DLManagedTensor.deleter", "deleter"], [0, 1, 1, "c.DLManagedTensor.dl_tensor", "dl_tensor"], [0, 1, 1, "c.DLManagedTensor.manager_ctx", "manager_ctx"]], "DLManagedTensorVersioned": [[0, 1, 1, "c.DLManagedTensorVersioned.deleter", "deleter"], [0, 1, 1, "c.DLManagedTensorVersioned.dl_tensor", "dl_tensor"], [0, 1, 1, "c.DLManagedTensorVersioned.flags", "flags"], [0, 1, 1, "c.DLManagedTensorVersioned.manager_ctx", "manager_ctx"], [0, 1, 1, "c.DLManagedTensorVersioned.version", "version"]], "DLPackVersion": [[0, 1, 1, "c.DLPackVersion.major", "major"], [0, 1, 1, "c.DLPackVersion.minor", "minor"]], "DLTensor": [[0, 1, 1, "c.DLTensor.byte_offset", "byte_offset"], [0, 1, 1, "c.DLTensor.data", "data"], [0, 1, 1, "c.DLTensor.device", "device"], [0, 1, 1, "c.DLTensor.dtype", "dtype"], [0, 1, 1, "c.DLTensor.ndim", "ndim"], [0, 1, 1, "c.DLTensor.shape", "shape"], [0, 1, 1, "c.DLTensor.strides", "strides"]]}, "objtypes": {"0": "c:struct", "1": "c:member", "2": "c:enum", "3": "c:enumerator", "4": "c:macro"}, "objnames": {"0": ["c", "struct", "C struct"], "1": ["c", "member", "C member"], "2": ["c", "enum", "C enum"], "3": ["c", "enumerator", "C enumerator"], "4": ["c", "macro", "C macro"]}, "titleterms": {"c": 0, "api": 0, "dlpack": [0, 1, 2], "h": 0, "macro": 0, "enumer": 0, "struct": 0, "welcom": 1, "": 1, "document": 1, "purpos": 1, "scope": 1, "roadmap": 1, "indic": 1, "tabl": 1, "python": 2, "specif": 2, "syntax": 2, "data": 2, "interchang": 2, "semant": 2, "implement": 2, "refer": 2}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"C API (dlpack.h)": [[0, "c-api-dlpack-h"]], "Macros": [[0, "macros"]], "Enumerations": [[0, "enumerations"]], "Structs": [[0, "structs"]], "Welcome to DLPack\u2019s documentation!": [[1, "welcome-to-dlpack-s-documentation"]], "Purpose": [[1, "purpose"]], "Scope": [[1, "scope"]], "Roadmap": [[1, "roadmap"]], "DLPack Documentation": [[1, "dlpack-documentation"]], "Indices and tables": [[1, "indices-and-tables"]], "Python Specification for DLPack": [[2, "python-specification-for-dlpack"]], "Syntax for data interchange with DLPack": [[2, "syntax-for-data-interchange-with-dlpack"]], "Semantics": [[2, "semantics"]], "Implementation": [[2, "implementation"]], "Reference Implementations": [[2, "reference-implementations"]]}, "indexentries": {"dldatatype (c struct)": [[0, "c.DLDataType"]], "dldatatype.bits (c var)": [[0, "c.DLDataType.bits"]], "dldatatype.code (c var)": [[0, "c.DLDataType.code"]], "dldatatype.lanes (c var)": [[0, "c.DLDataType.lanes"]], "dldatatypecode (c enum)": [[0, "c.DLDataTypeCode"]], "dldatatypecode.kdlbfloat (c enumerator)": [[0, "c.DLDataTypeCode.kDLBfloat"]], "dldatatypecode.kdlbool (c enumerator)": [[0, "c.DLDataTypeCode.kDLBool"]], "dldatatypecode.kdlcomplex (c enumerator)": [[0, "c.DLDataTypeCode.kDLComplex"]], "dldatatypecode.kdlfloat (c enumerator)": [[0, "c.DLDataTypeCode.kDLFloat"]], "dldatatypecode.kdlint (c enumerator)": [[0, "c.DLDataTypeCode.kDLInt"]], "dldatatypecode.kdlopaquehandle (c enumerator)": [[0, "c.DLDataTypeCode.kDLOpaqueHandle"]], "dldatatypecode.kdluint (c enumerator)": [[0, "c.DLDataTypeCode.kDLUInt"]], "dldevice (c struct)": [[0, "c.DLDevice"]], "dldevice.device_id (c var)": [[0, "c.DLDevice.device_id"]], "dldevice.device_type (c var)": [[0, "c.DLDevice.device_type"]], "dldevicetype (c enum)": [[0, "c.DLDeviceType"]], "dldevicetype.kdlcpu (c enumerator)": [[0, "c.DLDeviceType.kDLCPU"]], "dldevicetype.kdlcuda (c enumerator)": [[0, "c.DLDeviceType.kDLCUDA"]], "dldevicetype.kdlcudahost (c enumerator)": [[0, "c.DLDeviceType.kDLCUDAHost"]], "dldevicetype.kdlcudamanaged (c enumerator)": [[0, "c.DLDeviceType.kDLCUDAManaged"]], "dldevicetype.kdlextdev (c enumerator)": [[0, "c.DLDeviceType.kDLExtDev"]], "dldevicetype.kdlhexagon (c enumerator)": [[0, "c.DLDeviceType.kDLHexagon"]], "dldevicetype.kdlmaia (c enumerator)": [[0, "c.DLDeviceType.kDLMAIA"]], "dldevicetype.kdlmetal (c enumerator)": [[0, "c.DLDeviceType.kDLMetal"]], "dldevicetype.kdloneapi (c enumerator)": [[0, "c.DLDeviceType.kDLOneAPI"]], "dldevicetype.kdlopencl (c enumerator)": [[0, "c.DLDeviceType.kDLOpenCL"]], "dldevicetype.kdlrocm (c enumerator)": [[0, "c.DLDeviceType.kDLROCM"]], "dldevicetype.kdlrocmhost (c enumerator)": [[0, "c.DLDeviceType.kDLROCMHost"]], "dldevicetype.kdlvpi (c enumerator)": [[0, "c.DLDeviceType.kDLVPI"]], "dldevicetype.kdlvulkan (c enumerator)": [[0, "c.DLDeviceType.kDLVulkan"]], "dldevicetype.kdlwebgpu (c enumerator)": [[0, "c.DLDeviceType.kDLWebGPU"]], "dlmanagedtensor (c struct)": [[0, "c.DLManagedTensor"]], "dlmanagedtensor.deleter (c var)": [[0, "c.DLManagedTensor.deleter"]], "dlmanagedtensor.dl_tensor (c var)": [[0, "c.DLManagedTensor.dl_tensor"]], "dlmanagedtensor.manager_ctx (c var)": [[0, "c.DLManagedTensor.manager_ctx"]], "dlmanagedtensorversioned (c struct)": [[0, "c.DLManagedTensorVersioned"]], "dlmanagedtensorversioned.deleter (c var)": [[0, "c.DLManagedTensorVersioned.deleter"]], "dlmanagedtensorversioned.dl_tensor (c var)": [[0, "c.DLManagedTensorVersioned.dl_tensor"]], "dlmanagedtensorversioned.flags (c var)": [[0, "c.DLManagedTensorVersioned.flags"]], "dlmanagedtensorversioned.manager_ctx (c var)": [[0, "c.DLManagedTensorVersioned.manager_ctx"]], "dlmanagedtensorversioned.version (c var)": [[0, "c.DLManagedTensorVersioned.version"]], "dlpack_dll (c macro)": [[0, "c.DLPACK_DLL"]], "dlpack_extern_c (c macro)": [[0, "c.DLPACK_EXTERN_C"]], "dlpack_flag_bitmask_is_copied (c macro)": [[0, "c.DLPACK_FLAG_BITMASK_IS_COPIED"]], "dlpack_flag_bitmask_read_only (c macro)": [[0, "c.DLPACK_FLAG_BITMASK_READ_ONLY"]], "dlpack_major_version (c macro)": [[0, "c.DLPACK_MAJOR_VERSION"]], "dlpack_minor_version (c macro)": [[0, "c.DLPACK_MINOR_VERSION"]], "dlpackversion (c struct)": [[0, "c.DLPackVersion"]], "dlpackversion.major (c var)": [[0, "c.DLPackVersion.major"]], "dlpackversion.minor (c var)": [[0, "c.DLPackVersion.minor"]], "dltensor (c struct)": [[0, "c.DLTensor"]], "dltensor.byte_offset (c var)": [[0, "c.DLTensor.byte_offset"]], "dltensor.data (c var)": [[0, "c.DLTensor.data"]], "dltensor.device (c var)": [[0, "c.DLTensor.device"]], "dltensor.dtype (c var)": [[0, "c.DLTensor.dtype"]], "dltensor.ndim (c var)": [[0, "c.DLTensor.ndim"]], "dltensor.shape (c var)": [[0, "c.DLTensor.shape"]], "dltensor.strides (c var)": [[0, "c.DLTensor.strides"]]}})