"""更深入地检查 NVENC API。"""
import ctypes
import os
import sys

print(f"Python: {sys.version}")
print(f"Arch: {'64-bit' if sys.maxsize > 2**32 else '32-bit'}")

dll = ctypes.cdll.LoadLibrary('nvEncodeAPI64.dll')
func = dll.NvEncodeAPICreateInstance
func.restype = ctypes.c_uint32

# The NVENC API function list struct for older versions
# Let's try to define a minimal but correct struct

# NVENC API version constants from nvEncodeAPI.h
# The version is encoded as: (major << 24) | (minor << 16) | (patch << 8) | build
# NV_ENC_API_VERSION = 0x000C0000 (12.0)

# But the struct version field might be different
# NV_ENCODE_API_FUNCTION_LIST_VER = 0x000C0000 (12.0)

# Let's try creating the struct properly
class NVENCAPI_FUNCTION_LIST(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),           # 0x000C0000
        ('reserved', ctypes.c_uint32 * 8),       # 32 bytes
        # Function pointers (all void* for now)
        ('nvEncOpenEncodeSession', ctypes.c_void_p),
        ('nvEncOpenEncodeSessionEx', ctypes.c_void_p),
        ('nvEncGetEncodeGUIDCount', ctypes.c_void_p),
        ('nvEncGetEncodeProfileGUIDCount', ctypes.c_void_p),
        ('nvEncGetEncodeGUIDs', ctypes.c_void_p),
        ('nvEncGetEncodeProfileGUIDs', ctypes.c_void_p),
        ('nvEncGetInputFormats', ctypes.c_void_p),
        ('nvEncGetEncodeCaps', ctypes.c_void_p),
        ('nvEncGetEncodePresetCount', ctypes.c_void_p),
        ('nvEncGetEncodePresetGUIDs', ctypes.c_void_p),
        ('nvEncGetEncodePresetConfig', ctypes.c_void_p),
        ('nvEncInitializeEncoder', ctypes.c_void_p),
        ('nvEncCreateInputBuffer', ctypes.c_void_p),
        ('nvEncDestroyInputBuffer', ctypes.c_void_p),
        ('nvEncCreateBitstreamBuffer', ctypes.c_void_p),
        ('nvEncDestroyBitstreamBuffer', ctypes.c_void_p),
        ('nvEncLockBitstream', ctypes.c_void_p),
        ('nvEncUnlockBitstream', ctypes.c_void_p),
        ('nvEncLockInputBuffer', ctypes.c_void_p),
        ('nvEncUnlockInputBuffer', ctypes.c_void_p),
        ('nvEncEncodePicture', ctypes.c_void_p),
        ('nvEncFlushEncoderQueue', ctypes.c_void_p),
        ('nvEncGetSequenceParam', ctypes.c_void_p),
        ('nvEncRegisterResource', ctypes.c_void_p),
        ('nvEncUnregisterResource', ctypes.c_void_p),
        ('nvEncMapInputResource', ctypes.c_void_p),
        ('nvEncUnmapInputResource', ctypes.c_void_p),
        ('nvEncDestroyEncoder', ctypes.c_void_p),
        ('nvEncInvalidateRefFrames', ctypes.c_void_p),
        ('nvEncOpenEncodeSessionEx_2', ctypes.c_void_p),
        ('nvEncRegisterResourceEx', ctypes.c_void_p),
        ('nvEncGetEncoderInfo', ctypes.c_void_p),
        ('nvEncGetStatistics', ctypes.c_void_p),
        ('nvEncGetSequenceH264', ctypes.c_void_p),
        ('nvEncGetSequenceHEVC', ctypes.c_void_p),
        # Additional pointers for newer API versions
        ('nvEncGetSequenceAV1', ctypes.c_void_p),
        ('nvEncRegisterAsyncEvent', ctypes.c_void_p),
        ('nvEncUnregisterAsyncEvent', ctypes.c_void_p),
        ('nvEncWaitForEvent', ctypes.c_void_p),
        ('nvEncFlushEncoderQueue2', ctypes.c_void_p),
    ]

print(f"\nStruct size: {ctypes.sizeof(NVENCAPI_FUNCTION_LIST)} bytes")

# Try with properly defined struct
api_list = NVENCAPI_FUNCTION_LIST()
ctypes.memset(ctypes.addressof(api_list), 0, ctypes.sizeof(api_list))

# Try different version values
for ver in [0x000C0000, 0x000C0001, 0x000C0002, 0x000C0003, 0x000D0000, 0x000E0000]:
    api_list.version = ver
    ret = func(ctypes.byref(api_list))
    if ret == 0:
        print(f"Version 0x{ver:08X}: SUCCESS!")
        # List all non-null function pointers
        for name, _ in NVENCAPI_FUNCTION_LIST._fields_:
            val = getattr(api_list, name)
            if val is not None and name != 'version' and not name.startswith('reserved'):
                print(f"  {name}: {hex(val.value if isinstance(val, ctypes.c_void_p) else val)}")
        break
    else:
        print(f"Version 0x{ver:08X}: ret={ret} (0x{ret:08X})")

# If all failed, let's try with a simpler struct
print(f"\nTrying with minimal struct...")
class MinimalStruct(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 8),
        ('fns', ctypes.c_void_p * 50),
    ]

print(f"Minimal struct size: {ctypes.sizeof(MinimalStruct)} bytes")

for ver in [0x000C0000, 0x000C0001, 0x000C0002, 0x000C0003, 0x000D0000, 0x000E0000]:
    ms = MinimalStruct()
    ctypes.memset(ctypes.addressof(ms), 0, ctypes.sizeof(ms))
    ms.version = ver
    ret = func(ctypes.byref(ms))
    if ret == 0:
        print(f"  Version 0x{ver:08X}: SUCCESS!")
        for i in range(50):
            if ms.fns[i] is not None:
                print(f"    fn[{i}]: {hex(ms.fns[i])}")
        break
    else:
        print(f"  Version 0x{ver:08X}: ret={ret}")