"""尝试不同方式调用 NVENC API。"""
import ctypes
import os

dll = ctypes.cdll.LoadLibrary('nvEncodeAPI64.dll')
func = dll.NvEncodeAPICreateInstance
func.restype = ctypes.c_uint32

# Try 1: Direct struct with correct version
# NV_ENCODE_API_FUNCTION_LIST_VER = 12.0
NV_ENC_API_VER_V12 = 0x000C0000
NV_ENC_API_VER_V11 = 0x000B0000
NV_ENC_API_VER_V10 = 0x000A0000

# Let's try different struct sizes
# The struct for API 12.0 has:
# 1 uint32 version + 8 uint32 reserved + N function pointers
# Function pointers on 64-bit = 8 bytes each

# Let's count how many function pointers are in the struct
# From the NVENC SDK header, API 12.0 has about 35 function pointers
# Total size = 4 + 32 + 35*8 = 316 bytes

# But let's try a much larger buffer to be safe
for api_ver in [NV_ENC_API_VER_V12, NV_ENC_API_VER_V11, NV_ENC_API_VER_V10]:
    # Create a buffer large enough for the struct
    buf_size = 4096
    buf = (ctypes.c_uint8 * buf_size)()
    ctypes.memset(buf, 0, buf_size)
    
    # Set version field (first 4 bytes)
    version_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))
    version_ptr[0] = api_ver
    
    # Call the function
    ret = func(buf)
    print(f"API 0x{api_ver:08X}: ret={ret}")
    
    if ret == 0:
        print(f"  SUCCESS!")
        # Read the function pointers
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        print(f"  Pointer size: {ptr_size} bytes")
        
        # List all non-null function pointers
        for i in range(50):
            offset = 4 + 32 + i * ptr_size  # version + reserved
            if offset + ptr_size > buf_size:
                break
            ptr = ctypes.c_void_p.from_buffer(buf, offset)
            if ptr.value is not None:
                print(f"  fn[{i}] at offset {offset}: {hex(ptr.value)}")
        break

# If all failed, let's try the version from the DLL
print(f"\nTrying with different version schemes...")
# The DLL version 32.0.992676
# NVENC API version is typically encoded as:
# major << 24 | minor << 16 | patch << 8 | build
# 32.0 = 0x00200000
for api_ver in [0x00200000, 0x00140000, 0x00120000, 0x000C0000, 0x000B0000, 0x000A0000, 0x00000001]:
    buf = (ctypes.c_uint8 * 4096)()
    ctypes.memset(buf, 0, 4096)
    version_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))
    version_ptr[0] = api_ver
    ret = func(buf)
    if ret == 0:
        print(f"  API 0x{api_ver:08X}: SUCCESS!")
        break
    else:
        print(f"  API 0x{api_ver:08X}: ret={ret}")