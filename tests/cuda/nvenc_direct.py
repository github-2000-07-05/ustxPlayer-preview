"""NVENC 零拷贝编码器 — 通过 ctypes 直接调用 nvEncodeAPI64.dll。

核心思路：
    CUDA 渲染（CuPy）和 NVENC 编码在同一张 GPU 上，显存共享。
    渲染后的帧数据（NV12）在 GPU 显存中，通过注册 CUDA 设备指针
    给 NVENC 直接编码，数据不出 GPU，实现真正的零拷贝。

    本模块通过 ctypes 封装 nvEncodeAPI64.dll（NVIDIA 驱动自带，
    无需额外安装 SDK），实现 H.264 NVENC 编码。

使用流程：
    encoder = NvencEncoder(width, height, fps, bitrate)
    encoder.open(cuda_context)          # 传入 CuPy 的 CUDA 上下文
    encoder.register_buffer(device_ptr) # 注册 CuPy 数组的设备指针
    for each frame:
        # 渲染到 CuPy 数组（NV12 格式）
        bitstream = encoder.encode()
        output_file.write(bitstream)
    encoder.flush()                     # 发送 EOS
    encoder.close()

依赖：
    - NVIDIA 驱动（含 nvEncodeAPI64.dll）
    - cupy（用于 CUDA 渲染和内存管理）
"""

import ctypes
import ctypes.util
import os
import sys
from typing import Optional, List, Tuple


# =============================================================================
# NVENC API 常量
# =============================================================================

# API 版本号
NV_ENC_API_VER = 0x000C0000  # 12.0
NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = 0x00000001
NV_ENC_INITIALIZE_PARAMS_VER = 0x00000005  # API 12.0+
NV_ENC_CONFIG_VER = 0x0000000C  # API 12.0+
NV_ENC_PRESET_CONFIG_VER = 0x00000001
NV_ENC_PIC_PARAMS_VER = 0x0000000B  # API 12.1+
NV_ENC_REGISTER_RESOURCE_VER = 0x00000002  # API 12.0+
NV_ENC_MAP_INPUT_RESOURCE_VER = 0x00000001
NV_ENC_LOCK_BITSTREAM_VER = 0x00000001
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = 0x00000001

# 设备类型
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_DEVICE_TYPE_D3D9 = 2
NV_ENC_DEVICE_TYPE_D3D11 = 3
NV_ENC_DEVICE_TYPE_VULKAN = 4

# 缓冲区格式
NV_ENC_BUFFER_FORMAT_NV12 = 0x00000004
NV_ENC_BUFFER_FORMAT_UNDEFINED = 0x00000000
NV_ENC_BUFFER_FORMAT_ARGB = 0x00000007
NV_ENC_BUFFER_FORMAT_ABGR = 0x00000010

# 输入资源类型
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 0x00000002

# 图片类型
NV_ENC_PIC_TYPE_IDR = 0x00000003
NV_ENC_PIC_TYPE_P = 0x00000000
NV_ENC_PIC_TYPE_B = 0x00000001
NV_ENC_PIC_TYPE_SKIP = 0x00000004

# 编码器 GUID
NV_ENC_CODEC_H264_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x41, 0x43, 0x45, 0x4E, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_CODEC_HEVC_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x45, 0x56, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

# H.264 预设 GUID
NV_ENC_PRESET_P1_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P2_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P3_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P4_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P5_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x35, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P6_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
NV_ENC_PRESET_P7_GUID = bytes([0x1F, 0x7A, 0x56, 0x48, 0x50, 0x37, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

# 预设配置
NV_ENC_TWO_PASS_QUARTER = 0x00000001
NV_ENC_TWO_PASS_DISABLED = 0x00000000

NV_ENC_RC_MODE_CONSTQP = 0x00000000
NV_ENC_RC_MODE_VBR = 0x00000001
NV_ENC_RC_MODE_CBR = 0x00000002
NV_ENC_RC_MODE_CBR_LOWDELAY_HQ = 0x00000008
NV_ENC_RC_MODE_CBR_HQ = 0x00000010
NV_ENC_RC_MODE_VBR_HQ = 0x00000020

# 状态码
NV_ENC_SUCCESS = 0x00000000
NV_ENC_ERR_INVALID_ENCODER = 0x00000001
NV_ENC_ERR_INVALID_DEVICE = 0x00000002
NV_ENC_ERR_INVALID_ENCODERDEVICE = 0x00000003
NV_ENC_ERR_INVALID_VERSION = 0x00000004
NV_ENC_ERR_OUT_OF_MEMORY = 0x00000005
NV_ENC_ERR_GENERIC = 0x00000006
NV_ENC_ERR_UNIMPLEMENTED = 0x00000007
NV_ENC_ERR_INVALID_PARAM = 0x00000008
NV_ENC_ERR_INVALID_PTR = 0x00000009
NV_ENC_ERR_INVALID_CALL = 0x0000000A


# =============================================================================
# NVENC API 结构体定义
# =============================================================================

class GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', ctypes.c_uint32),
        ('Data2', ctypes.c_uint16),
        ('Data3', ctypes.c_uint16),
        ('Data4', ctypes.c_uint8 * 8),
    ]


def make_guid(data: bytes) -> GUID:
    """从 16 字节数据创建 GUID。"""
    assert len(data) == 16
    d1 = int.from_bytes(data[0:4], 'little')
    d2 = int.from_bytes(data[4:6], 'little')
    d3 = int.from_bytes(data[6:8], 'little')
    d4 = (ctypes.c_uint8 * 8)(*data[8:16])
    return GUID(d1, d2, d3, d4)


# =============================================================================
# NVENC API 结构体
# =============================================================================

class NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('device', ctypes.c_void_p),
        ('deviceType', ctypes.c_uint32),
        ('apiVersion', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 8),
    ]


class NV_ENC_CONFIG_H264(ctypes.Structure):
    """H.264 编码配置（简化版，仅设置必要字段）。"""
    _fields_ = [
        ('enableFMO', ctypes.c_uint32),
        ('enableASO', ctypes.c_uint32),
        ('enableSliceEncoding', ctypes.c_uint32),
        ('enableConstrainedEncoding', ctypes.c_uint32),
        ('enableIntraRefresh', ctypes.c_uint32),
        ('enableBFrames', ctypes.c_uint32),
        ('qpPrimer', ctypes.c_uint32),
        ('weightedPred', ctypes.c_uint32),
        ('weightedBiPred', ctypes.c_uint32),
        ('level', ctypes.c_uint32),
        ('idrPeriod', ctypes.c_uint32),
        ('separateColourPlaneFlag', ctypes.c_uint32),
        ('chromaFormatIDC', ctypes.c_uint32),
        ('adaptiveTransformMode', ctypes.c_uint32),
        ('fmoMode', ctypes.c_uint32),
        ('numSliceGroupMinus1', ctypes.c_uint32),
        ('numSliceGroups', ctypes.c_uint32),
        ('sliceGroupMapType', ctypes.c_uint32),
        ('sliceGroupChangeRate', ctypes.c_uint32),
        ('sliceGroupMap', ctypes.c_uint32 * 64),
        ('sliceGroupMapEntry', ctypes.c_uint32),
        ('sliceGroupMapEntryCount', ctypes.c_uint32),
        ('numRefFramesInPeriod', ctypes.c_uint32),
        ('slice_cabac_flag', ctypes.c_uint32),
        ('deblockingFilter', ctypes.c_uint32),
        ('disableDeblockingFilterIDC', ctypes.c_uint32),
        ('sliceActDeltaQpEnableFlag', ctypes.c_uint32),
        ('sliceActDeltaQpThresholdArray', ctypes.c_uint32 * 16),
        ('sliceActDeltaQpArray', ctypes.c_uint32 * 16),
        ('useMBBRC', ctypes.c_uint32),
        ('enableIntra64x64', ctypes.c_uint32),
        ('enableIntra32x32', ctypes.c_uint32),
        ('enableIntra16x16', ctypes.c_uint32),
        ('enableIntra8x8', ctypes.c_uint32),
        ('enableIntra4x4', ctypes.c_uint32),
        ('enableIntra16x16Planar', ctypes.c_uint32),
        ('enableIntra16x16DC', ctypes.c_uint32),
        ('enableIntra16x16Horizontal', ctypes.c_uint32),
        ('enableIntra16x16Vertical', ctypes.c_uint32),
        ('enableIntra8x8Planar', ctypes.c_uint32),
        ('enableIntra8x8DC', ctypes.c_uint32),
        ('enableIntra8x8Horizontal', ctypes.c_uint32),
        ('enableIntra8x8Vertical', ctypes.c_uint32),
        ('enableIntra4x4Planar', ctypes.c_uint32),
        ('enableIntra4x4DC', ctypes.c_uint32),
        ('enableIntra4x4Horizontal', ctypes.c_uint32),
        ('enableIntra4x4Vertical', ctypes.c_uint32),
        ('intra4x4Dir', ctypes.c_uint32 * 16),
        ('enableIntra8x8Dir', ctypes.c_uint32 * 64),
        ('enableIntra16x16Dir', ctypes.c_uint32 * 16),
        ('enableBiPred', ctypes.c_uint32),
        ('enableWeightedBiPred', ctypes.c_uint32),
        ('enableDirect8x8Inference', ctypes.c_uint32),
        ('enableDirectSpatialTemporal', ctypes.c_uint32),
        ('constraint_set0_flag', ctypes.c_uint32),
        ('constraint_set1_flag', ctypes.c_uint32),
        ('constraint_set2_flag', ctypes.c_uint32),
        ('constraint_set3_flag', ctypes.c_uint32),
        ('constraint_set4_flag', ctypes.c_uint32),
        ('constraint_set5_flag', ctypes.c_uint32),
        ('enableSPS_override', ctypes.c_uint32),
        ('enablePPS_override', ctypes.c_uint32),
        ('refPicMarkRepDisable', ctypes.c_uint32),
        ('qvbrQuality', ctypes.c_uint32),
        ('numTemporalLayers', ctypes.c_uint32),
        ('enableMinigop', ctypes.c_uint32),
        ('fieldEncoding', ctypes.c_uint32),
        ('disableSPSPPS', ctypes.c_uint32),
        ('repeatSPSPPS', ctypes.c_uint32),
        ('enableIntraRefresh', ctypes.c_uint32),
        ('intraRefreshPeriod', ctypes.c_uint32),
        ('intraRefreshDur', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_CONFIG(ctypes.Structure):
    """编码配置结构体。"""
    pass


class NV_ENC_CONFIG_HEVC(ctypes.Structure):
    _fields_ = [
        ('level', ctypes.c_uint32),
        ('tier', ctypes.c_uint32),
        ('minCUSize', ctypes.c_uint32),
        ('maxCUSize', ctypes.c_uint32),
        ('minTUSize', ctypes.c_uint32),
        ('maxTUSize', ctypes.c_uint32),
        ('maxTrDepth', ctypes.c_uint32),
        ('trDepthIntra', ctypes.c_uint32),
        ('trDepthInter', ctypes.c_uint32),
        ('maxNumRefFrames', ctypes.c_uint32),
        ('useMBBRC', ctypes.c_uint32),
        ('enableIntraRefresh', ctypes.c_uint32),
        ('intraRefreshPeriod', ctypes.c_uint32),
        ('intraRefreshDur', ctypes.c_uint32),
        ('numTemporalLayers', ctypes.c_uint32),
        ('enableTemporalAQ', ctypes.c_uint32),
        ('enableScenecut', ctypes.c_uint32),
        ('enableConstrainedEncoding', ctypes.c_uint32),
        ('enableLookAhead', ctypes.c_uint32),
        ('disableSPSPPS', ctypes.c_uint32),
        ('repeatSPSPPS', ctypes.c_uint32),
        ('enableIntra64x64', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_CONFIG_H264_SPS_PPS(ctypes.Structure):
    _fields_ = [
        ('sps', ctypes.c_uint8 * 128),
        ('spsLength', ctypes.c_uint32),
        ('pps', ctypes.c_uint8 * 128),
        ('ppsLength', ctypes.c_uint32),
    ]


# 前向声明完成
NV_ENC_CONFIG._fields_ = [
    ('version', ctypes.c_uint32),
    ('profile', ctypes.c_uint32),
    ('level', ctypes.c_uint32),
    ('profileGUID', GUID),
    ('bufferFormat', ctypes.c_uint32),
    ('motionEstimationPrecision', ctypes.c_uint32),
    ('gopLength', ctypes.c_uint32),
    ('frameIntervalP', ctypes.c_uint32),
    ('monoChrome', ctypes.c_uint32),
    ('frameFieldMode', ctypes.c_uint32),
    ('mvPrecision', ctypes.c_uint32),
    ('encodeWidth', ctypes.c_uint32),
    ('encodeHeight', ctypes.c_uint32),
    ('darWidth', ctypes.c_uint32),
    ('darHeight', ctypes.c_uint32),
    ('frameRateNum', ctypes.c_uint32),
    ('frameRateDen', ctypes.c_uint32),
    ('enablePTD', ctypes.c_uint32),
    ('reportSliceOffsets', ctypes.c_uint32),
    ('enableSubFrameWrite', ctypes.c_uint32),
    ('enableExternalMEHints', ctypes.c_uint32),
    ('enableMEOnlyMode', ctypes.c_uint32),
    ('enableWeightedPrediction', ctypes.c_uint32),
    ('rcMode', ctypes.c_uint32),
    ('qp', ctypes.c_uint32),
    ('bitRate', ctypes.c_uint32),
    ('maxBitRate', ctypes.c_uint32),
    ('vbvBufferSize', ctypes.c_uint32),
    ('vbvInitialDelay', ctypes.c_uint32),
    ('targetBufferSize', ctypes.c_uint32),
    ('targetQuality', ctypes.c_uint32),
    ('targetQualityLSB', ctypes.c_uint32),
    ('lowDelayKeyFrameScale', ctypes.c_uint32),
    ('enableLookAhead', ctypes.c_uint32),
    ('lookAheadDepth', ctypes.c_uint32),
    ('presetGUID', GUID),
    ('codec', ctypes.c_uint32),
    ('avgQP', ctypes.c_uint32),
    ('frameQuality', ctypes.c_uint32),
    ('presetGUID2', GUID),
    ('reserved', ctypes.c_uint32 * 288),
    # Codec-specific config (union, we use H.264)
    ('encodeCodecConfig', ctypes.c_uint8 * 2048),  # 大块预分配
]


class NV_ENC_PRESET_CONFIG(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('presetCfg', NV_ENC_CONFIG),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_INITIALIZE_PARAMS(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('encodeGUID', GUID),
        ('presetGUID', GUID),
        ('encodeWidth', ctypes.c_uint32),
        ('encodeHeight', ctypes.c_uint32),
        ('darWidth', ctypes.c_uint32),
        ('darHeight', ctypes.c_uint32),
        ('frameRateNum', ctypes.c_uint32),
        ('frameRateDen', ctypes.c_uint32),
        ('enableEncodeAsync', ctypes.c_uint32),
        ('reportSliceOffsets', ctypes.c_uint32),
        ('enableSubFrameWrite', ctypes.c_uint32),
        ('enableExternalMEHints', ctypes.c_uint32),
        ('enableMEOnlyMode', ctypes.c_uint32),
        ('enableWeightedPrediction', ctypes.c_uint32),
        ('maxEncodeWidth', ctypes.c_uint32),
        ('maxEncodeHeight', ctypes.c_uint32),
        ('maxMEHintCountsPerBlock', ctypes.c_uint32 * 2),
        ('enableEncodeAsync_2', ctypes.c_uint32),
        ('enablePTD', ctypes.c_uint32),
        ('enableOutputIncompleteFrames', ctypes.c_uint32),
        ('enableUserSEI', ctypes.c_uint32),
        ('enableTemporalAQ', ctypes.c_uint32),
        ('enableFillerDataInsertion', ctypes.c_uint32),
        ('enableLtr', ctypes.c_uint32),
        ('enableLossless', ctypes.c_uint32),
        ('numMEHintCountsPerBlock', ctypes.c_uint32 * 2),
        ('enableHevcLossless', ctypes.c_uint32),
        ('enableQualityMode', ctypes.c_uint32),
        ('enableOutputVideoAfterEncode', ctypes.c_uint32),
        ('enableFieldMode', ctypes.c_uint32),
        ('enableNoScenecapDetection', ctypes.c_uint32),
        ('enablePrependSPSPPSToIDR', ctypes.c_uint32),
        ('enableMultiPassEncoding', ctypes.c_uint32),
        ('tuningInfo', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 256),
        # 编码配置（指针）
        ('encodeConfig', ctypes.POINTER(NV_ENC_CONFIG)),
        ('reserved2', ctypes.c_void_p * 64),
    ]


class NV_ENC_PIC_PARAMS(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('inputWidth', ctypes.c_uint32),
        ('inputHeight', ctypes.c_uint32),
        ('inputPitch', ctypes.c_uint32),
        ('encodePicFlags', ctypes.c_uint32),
        ('frameIdx', ctypes.c_uint32),
        ('inputDuration', ctypes.c_uint64),
        ('inputTimestamp', ctypes.c_uint64),
        ('inputBuffer', ctypes.c_void_p),
        ('outputBitstream', ctypes.c_void_p),
        ('bufferFmt', ctypes.c_uint32),
        ('pictureStruct', ctypes.c_uint32),
        ('pictureType', ctypes.c_uint32),
        ('bRefMode', ctypes.c_uint32),
        ('qpDeltaMap', ctypes.c_void_p),
        ('qpDeltaMapSize', ctypes.c_uint32),
        ('codecPicParams', ctypes.c_void_p),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_REGISTER_RESOURCE(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('resourceType', ctypes.c_uint32),
        ('width', ctypes.c_uint32),
        ('height', ctypes.c_uint32),
        ('pitch', ctypes.c_uint32),
        ('subResourceIndex', ctypes.c_uint32),
        ('resourceToRegister', ctypes.c_void_p),  # CUdeviceptr
        ('registeredResource', ctypes.c_void_p),
        ('bufferFormat', ctypes.c_uint32),
        ('bufferUsage', ctypes.c_uint32),
        ('reserved', ctypes.c_void_p * 8),
    ]


class NV_ENC_MAP_INPUT_RESOURCE(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('registeredResource', ctypes.c_void_p),
        ('mappedResource', ctypes.c_void_p),
        ('mappedBufferFmt', ctypes.c_uint32),
        ('inputWidth', ctypes.c_uint32),
        ('inputHeight', ctypes.c_uint32),
        ('inputPitch', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_LOCK_BITSTREAM(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('outputBitstream', ctypes.c_void_p),
        ('bitstreamBufferPtr', ctypes.c_void_p),
        ('bitstreamSizeInBytes', ctypes.c_uint32),
        ('bitstreamSizeInBytes_2', ctypes.c_uint32),
        ('outputTimeStamp', ctypes.c_uint64),
        ('pictureType', ctypes.c_uint32),
        ('frameIdx', ctypes.c_uint32),
        ('hwEncodeStatus', ctypes.c_uint32),
        ('maxOutputSize', ctypes.c_uint32),
        ('avgQP', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 256),
    ]


class NV_ENC_CREATE_BITSTREAM_BUFFER(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('size', ctypes.c_uint32),
        ('bitstreamBuffer', ctypes.c_void_p),
        ('bitstreamBufferPtr', ctypes.c_void_p),
        ('reserved', ctypes.c_uint32 * 256),
    ]


# =============================================================================
# NVENC API 函数表
# =============================================================================

class NV_ENCODE_API_FUNCTION_LIST(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 8),
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
    ]


# =============================================================================
# NVENC 编码器类
# =============================================================================

_NVENC_DLL = None
_NVENC_API = None


def _load_nvenc() -> ctypes.CDLL:
    """加载 nvEncodeAPI64.dll（NVIDIA 驱动自带）。"""
    global _NVENC_DLL
    if _NVENC_DLL is not None:
        return _NVENC_DLL

    # 查找 nvEncodeAPI64.dll
    dll_path = ctypes.util.find_library("nvEncodeAPI64")
    if dll_path is None:
        # 尝试常见路径
        search_paths = [
            "nvEncodeAPI64.dll",
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32", "nvEncodeAPI64.dll"),
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "SysWOW64", "nvEncodeAPI64.dll"),
        ]
        for p in search_paths:
            if os.path.exists(p):
                dll_path = p
                break

    if dll_path is None:
        raise RuntimeError("未找到 nvEncodeAPI64.dll。请确保已安装 NVIDIA 驱动。")

    _NVENC_DLL = ctypes.cdll.LoadLibrary(dll_path)
    return _NVENC_DLL


def _get_api() -> NV_ENCODE_API_FUNCTION_LIST:
    """获取 NVENC API 函数表。"""
    global _NVENC_API
    if _NVENC_API is not None:
        return _NVENC_API

    dll = _load_nvenc()

    # 创建函数表结构体
    api = NV_ENCODE_API_FUNCTION_LIST()
    api.version = NV_ENC_API_VER

    # 调用 NvEncodeAPICreateInstance 获取函数表
    func = dll.NvEncodeAPICreateInstance
    func.argtypes = [ctypes.POINTER(NV_ENCODE_API_FUNCTION_LIST)]
    func.restype = ctypes.c_uint32

    ret = func(ctypes.byref(api))
    if ret != NV_ENC_SUCCESS:
        raise RuntimeError(f"NvEncodeAPICreateInstance 失败: 错误码 {ret}")

    _NVENC_API = api
    return api


def _check_nvresult(ret: int, context: str = ""):
    """检查 NVENC 函数返回值，失败时抛出异常。"""
    if ret != NV_ENC_SUCCESS:
        err_names = {
            NV_ENC_ERR_INVALID_ENCODER: "NV_ENC_ERR_INVALID_ENCODER",
            NV_ENC_ERR_INVALID_DEVICE: "NV_ENC_ERR_INVALID_DEVICE",
            NV_ENC_ERR_INVALID_ENCODERDEVICE: "NV_ENC_ERR_INVALID_ENCODERDEVICE",
            NV_ENC_ERR_INVALID_VERSION: "NV_ENC_ERR_INVALID_VERSION",
            NV_ENC_ERR_OUT_OF_MEMORY: "NV_ENC_ERR_OUT_OF_MEMORY",
            NV_ENC_ERR_GENERIC: "NV_ENC_ERR_GENERIC",
            NV_ENC_ERR_UNIMPLEMENTED: "NV_ENC_ERR_UNIMPLEMENTED",
            NV_ENC_ERR_INVALID_PARAM: "NV_ENC_ERR_INVALID_PARAM",
            NV_ENC_ERR_INVALID_PTR: "NV_ENC_ERR_INVALID_PTR",
            NV_ENC_ERR_INVALID_CALL: "NV_ENC_ERR_INVALID_CALL",
        }
        err_name = err_names.get(ret, f"未知错误码 {ret}")
        msg = f"NVENC 错误 [{err_name}]"
        if context:
            msg += f" — {context}"
        raise RuntimeError(msg)


class NvencEncoder:
    """NVENC 零拷贝编码器。

    零拷贝原理：
        CUDA 和 NVENC 在同一 GPU 上共享显存。CuPy 渲染输出的 NV12 帧
        数据在 GPU 显存中，通过 nvEncRegisterResource 注册到 NVENC，
        NVENC 直接读取显存中的帧数据进行编码，数据不出 GPU。

    使用方式：
        encoder = NvencEncoder()
        encoder.open(cuda_context)  # cuda_context = cupy.cuda.Device(0).ctx
        encoder.init(width, height, fps, bitrate)
        encoder.register_buffer(device_ptr)  # CuPy 数组的 data.ptr

        for each frame:
            # 渲染到 CuPy 数组（NV12 格式）
            encoded_data = encoder.encode_frame()
            output_file.write(encoded_data)

        encoder.flush()
        encoder.close()
    """

    def __init__(self):
        self._api = _get_api()
        self._encoder = ctypes.c_void_p()
        self._initialized = False
        self._registered_resource = None  # type: Optional[ctypes.c_void_p]
        self._bitstream_buffer = None  # type: Optional[ctypes.c_void_p]
        self._width = 0
        self._height = 0
        self._fps = 0
        self._frame_count = 0
        self._config = None  # type: Optional[NV_ENC_CONFIG]

    def open(self, cuda_context: int) -> None:
        """打开 NVENC 编码会话。

        Args:
            cuda_context: CUDA 上下文句柄。从 CuPy 获取：
                >>> import cupy as cp
                >>> ctx = int(cp.cuda.Device(0).ctx)
        """
        # 打开编码会话
        open_params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        open_params.device = ctypes.c_void_p(cuda_context)
        open_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
        open_params.apiVersion = NV_ENC_API_VER

        # 获取函数指针
        nvEncOpenEncodeSessionEx = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.POINTER(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS),
            ctypes.POINTER(ctypes.c_void_p),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncOpenEncodeSessionEx)))

        ret = nvEncOpenEncodeSessionEx(
            ctypes.byref(open_params),
            ctypes.byref(self._encoder),
        )
        _check_nvresult(ret, "打开编码会话")

    def init(
        self,
        width: int,
        height: int,
        fps: int = 60,
        bitrate: int = 0,
        preset_guid: bytes = NV_ENC_PRESET_P1_GUID,
    ) -> None:
        """初始化编码器。

        Args:
            width: 视频宽度（必须是偶数）
            height: 视频高度（必须是偶数）
            fps: 帧率
            bitrate: 码率（bps）。0 表示使用默认值（constqp 模式）。
            preset_guid: NVENC 预设 GUID，默认为 P1（最快）
        """
        self._width = width
        self._height = height
        self._fps = fps

        # 创建编码配置
        config = NV_ENC_CONFIG()
        config.version = NV_ENC_CONFIG_VER
        config.profile = 0  # 自动选择
        config.profileGUID = make_guid(bytes([0x1F, 0x7A, 0x56, 0x48, 0x41, 0x56, 0x43, 0x68, 0x69, 0x67, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00]))  # 高配置
        config.gopLength = fps * 5  # 5 秒一个 IDR
        config.frameIntervalP = 1  # 只有 P 帧（无 B 帧）
        config.frameRateNum = fps  # 帧率分子
        config.frameRateDen = 1  # 帧率分母
        config.encodeWidth = width
        config.encodeHeight = height
        config.darWidth = width
        config.darHeight = height
        config.enablePTD = 1  # 启用 PTD
        config.rcMode = NV_ENC_RC_MODE_CONSTQP  # 恒定量化参数
        config.qp = 28  # QP 值
        config.bitRate = bitrate or (width * height * fps // 100)  # 默认码率
        config.maxBitRate = config.bitRate
        config.presetGUID = make_guid(preset_guid)
        config.codec = 0  # H.264

        # 设置 H.264 特定配置（通过 encodeCodecConfig 字段）
        # 使用 NV_ENC_CONFIG_H264 填充
        h264_config = NV_ENC_CONFIG_H264()
        h264_config.enableBFrames = 0  # 无 B 帧
        h264_config.level = 0  # 自动选择
        h264_config.idrPeriod = fps * 5
        h264_config.disableSPSPPS = 0
        h264_config.repeatSPSPPS = 1
        h264_config.slice_cabac_flag = 1
        h264_config.deblockingFilter = 1
        h264_config.useMBBRC = 0
        h264_config.numTemporalLayers = 0
        h264_config.enableIntraRefresh = 0

        # 复制 H.264 配置到 config.encodeCodecConfig
        h264_size = ctypes.sizeof(NV_ENC_CONFIG_H264)
        h264_buffer = (ctypes.c_uint8 * h264_size).from_address(
            ctypes.addressof(h264_config)
        )
        # 确保 encodeCodecConfig 足够大
        assert ctypes.sizeof(config.encodeCodecConfig) >= h264_size
        for i in range(h264_size):
            config.encodeCodecConfig[i] = h264_buffer[i]

        self._config = config

        # 初始化参数
        init_params = NV_ENC_INITIALIZE_PARAMS()
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = make_guid(NV_ENC_CODEC_H264_GUID)
        init_params.presetGUID = make_guid(preset_guid)
        init_params.encodeWidth = width
        init_params.encodeHeight = height
        init_params.darWidth = width
        init_params.darHeight = height
        init_params.frameRateNum = fps
        init_params.frameRateDen = 1
        init_params.enableEncodeAsync = 0  # 同步模式
        init_params.enablePTD = 1
        init_params.enableOutputIncompleteFrames = 0
        init_params.enableWeightedPrediction = 0
        init_params.encodeConfig = ctypes.pointer(config)

        # 调用 nvEncInitializeEncoder
        nvEncInitializeEncoder = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_INITIALIZE_PARAMS),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncInitializeEncoder)))

        ret = nvEncInitializeEncoder(self._encoder.value, ctypes.byref(init_params))
        _check_nvresult(ret, "初始化编码器")
        self._initialized = True

        # 创建输出比特流缓冲区
        self._create_bitstream_buffer()

    def _create_bitstream_buffer(self, size: int = 0) -> None:
        """创建输出比特流缓冲区。"""
        if size == 0:
            # 默认大小：足够容纳一帧 H.264 数据
            size = self._width * self._height * 2

        create_params = NV_ENC_CREATE_BITSTREAM_BUFFER()
        create_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
        create_params.size = size

        nvEncCreateBitstreamBuffer = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_CREATE_BITSTREAM_BUFFER),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncCreateBitstreamBuffer)))

        ret = nvEncCreateBitstreamBuffer(self._encoder.value, ctypes.byref(create_params))
        _check_nvresult(ret, "创建输出比特流缓冲区")

        self._bitstream_buffer = create_params.bitstreamBuffer

    def register_buffer(self, device_ptr: int, pitch: int = 0) -> None:
        """注册 CUDA 设备指针作为 NVENC 输入缓冲区。

        这是零拷贝的关键：将 CuPy 渲染的 NV12 帧数据所在的 GPU 显存地址
        注册给 NVENC，NVENC 直接从此地址读取数据编码，无需 CPU 中转。

        Args:
            device_ptr: CuPy 数组的设备指针（cupy_array.data.ptr）
            pitch: 行跨度（字节）。NV12 的 pitch = width（因为 Y 平面每像素 1 字节）。
                   0 表示自动计算（= width）。
        """
        if pitch == 0:
            pitch = self._width

        reg_params = NV_ENC_REGISTER_RESOURCE()
        reg_params.version = NV_ENC_REGISTER_RESOURCE_VER
        reg_params.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
        reg_params.width = self._width
        reg_params.height = self._height
        reg_params.pitch = pitch
        reg_params.subResourceIndex = 0
        reg_params.resourceToRegister = ctypes.c_void_p(device_ptr)
        reg_params.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12
        reg_params.bufferUsage = 0

        nvEncRegisterResource = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_REGISTER_RESOURCE),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncRegisterResource)))

        ret = nvEncRegisterResource(self._encoder.value, ctypes.byref(reg_params))
        _check_nvresult(ret, "注册输入缓冲区")
        self._registered_resource = reg_params.registeredResource

    def encode_frame(self) -> bytes:
        """编码一帧（零拷贝）。

        从已注册的 CUDA 缓冲区读取帧数据编码，输出 H.264 比特流。

        Returns:
            H.264 编码后的比特流字节
        """
        if not self._initialized:
            raise RuntimeError("编码器未初始化")

        # 映射输入资源
        map_params = NV_ENC_MAP_INPUT_RESOURCE()
        map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER
        map_params.registeredResource = self._registered_resource

        nvEncMapInputResource = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_MAP_INPUT_RESOURCE),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncMapInputResource)))

        ret = nvEncMapInputResource(self._encoder.value, ctypes.byref(map_params))
        _check_nvresult(ret, "映射输入资源")

        # 编码
        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.inputWidth = self._width
        pic_params.inputHeight = self._height
        pic_params.inputPitch = 0
        pic_params.encodePicFlags = 0
        pic_params.frameIdx = self._frame_count
        pic_params.inputDuration = 0
        pic_params.inputTimestamp = self._frame_count
        pic_params.inputBuffer = map_params.mappedResource
        pic_params.outputBitstream = self._bitstream_buffer
        pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12
        pic_params.pictureStruct = 0  # 逐行扫描
        pic_params.pictureType = NV_ENC_PIC_TYPE_IDR if self._frame_count == 0 else NV_ENC_PIC_TYPE_P

        nvEncEncodePicture = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_PIC_PARAMS),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncEncodePicture)))

        ret = nvEncEncodePicture(self._encoder.value, ctypes.byref(pic_params))
        _check_nvresult(ret, f"编码帧 {self._frame_count}")

        self._frame_count += 1

        # 取消映射输入资源
        nvEncUnmapInputResource = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncUnmapInputResource)))

        ret = nvEncUnmapInputResource(self._encoder.value, map_params.mappedResource)
        _check_nvresult(ret, "取消映射输入资源")

        # 锁定并读取比特流
        lock_params = NV_ENC_LOCK_BITSTREAM()
        lock_params.version = NV_ENC_LOCK_BITSTREAM_VER
        lock_params.outputBitstream = self._bitstream_buffer

        nvEncLockBitstream = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_LOCK_BITSTREAM),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncLockBitstream)))

        ret = nvEncLockBitstream(self._encoder.value, ctypes.byref(lock_params))
        _check_nvresult(ret, "锁定比特流")

        # 读取编码数据
        data_ptr = lock_params.bitstreamBufferPtr
        data_size = lock_params.bitstreamSizeInBytes
        if data_size > 0 and data_ptr:
            encoded_data = (ctypes.c_uint8 * data_size).from_address(
                ctypes.c_void_p(data_ptr).value
            )
            result = bytes(encoded_data)
        else:
            result = b""

        # 解锁比特流
        nvEncUnlockBitstream = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncUnlockBitstream)))

        ret = nvEncUnlockBitstream(self._encoder.value, self._bitstream_buffer)
        _check_nvresult(ret, "解锁比特流")

        return result

    def flush(self) -> List[bytes]:
        """刷新编码器缓冲区，获取所有剩余编码数据。

        Returns:
            剩余编码数据列表
        """
        if not self._initialized:
            return []

        # 发送 EOS
        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.encodePicFlags = 1  # EOS
        pic_params.inputBuffer = None
        pic_params.outputBitstream = self._bitstream_buffer

        nvEncEncodePicture = ctypes.CFUNCTYPE(
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(NV_ENC_PIC_PARAMS),
        )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncEncodePicture)))

        ret = nvEncEncodePicture(self._encoder.value, ctypes.byref(pic_params))
        _check_nvresult(ret, "刷新编码器 (EOS)")

        # 读取所有剩余数据
        results = []
        while True:
            lock_params = NV_ENC_LOCK_BITSTREAM()
            lock_params.version = NV_ENC_LOCK_BITSTREAM_VER
            lock_params.outputBitstream = self._bitstream_buffer

            nvEncLockBitstream = ctypes.CFUNCTYPE(
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.POINTER(NV_ENC_LOCK_BITSTREAM),
            )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncLockBitstream)))

            ret = nvEncLockBitstream(self._encoder.value, ctypes.byref(lock_params))
            if ret != NV_ENC_SUCCESS:
                break

            data_size = lock_params.bitstreamSizeInBytes
            data_ptr = lock_params.bitstreamBufferPtr
            if data_size > 0 and data_ptr:
                encoded_data = (ctypes.c_uint8 * data_size).from_address(
                    ctypes.c_void_p(data_ptr).value
                )
                results.append(bytes(encoded_data))

            nvEncUnlockBitstream = ctypes.CFUNCTYPE(
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncUnlockBitstream)))
            nvEncUnlockBitstream(self._encoder.value, self._bitstream_buffer)

        return results

    def close(self) -> None:
        """关闭编码器，释放所有资源。"""
        try:
            # 取消注册资源
            if self._registered_resource is not None:
                nvEncUnregisterResource = ctypes.CFUNCTYPE(
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncUnregisterResource)))
                try:
                    nvEncUnregisterResource(self._encoder.value, self._registered_resource)
                except Exception:
                    pass
                self._registered_resource = None

            # 销毁比特流缓冲区
            if self._bitstream_buffer is not None:
                nvEncDestroyBitstreamBuffer = ctypes.CFUNCTYPE(
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncDestroyBitstreamBuffer)))
                try:
                    nvEncDestroyBitstreamBuffer(self._encoder.value, self._bitstream_buffer)
                except Exception:
                    pass
                self._bitstream_buffer = None

            # 销毁编码器
            if self._encoder.value is not None:
                nvEncDestroyEncoder = ctypes.CFUNCTYPE(
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                )(ctypes.c_void_p(ctypes.addressof(self._api.nvEncDestroyEncoder)))
                try:
                    nvEncDestroyEncoder(self._encoder.value)
                except Exception:
                    pass
                self._encoder.value = None
        except Exception as e:
            print(f"关闭编码器时出错: {e}")
        finally:
            self._initialized = False


# =============================================================================
# 便利函数
# =============================================================================

def get_nvenc_device_ptr(cupy_array) -> int:
    """获取 CuPy 数组的设备指针。

    Args:
        cupy_array: CuPy 数组（必须是连续内存）

    Returns:
        设备指针值（CUdeviceptr）
    """
    return cupy_array.data.ptr


def check_nvenc_available() -> Tuple[bool, str]:
    """检查 NVENC 是否可用。

    Returns:
        (是否可用, 描述信息)
    """
    try:
        _load_nvenc()
        _get_api()
        return True, "NVENC 可用"
    except Exception as e:
        return False, f"NVENC 不可用: {e}"