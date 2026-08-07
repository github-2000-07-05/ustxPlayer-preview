/**
 * nvenc_encoder.cpp — NVENC 硬件编码器 (帧重复优化版)
 * ================================================
 *
 * 编译 (需要 NVENC SDK + CUDA Toolkit):
 *   cl /EHsc /std:c++17 nvenc_encoder.cpp /link nvencodeapi.lib cuda.lib
 *   或
 *   g++ nvenc_encoder.cpp -o nvenc_encoder.exe -lnvencodeapi -lcuda -ldl
 *
 * 依赖:
 *   - NVIDIA GPU (GTX 600 系+)
 *   - NVIDIA 显卡驱动 (包含 nvEncodeAPI64.dll)
 *   - CUDA Toolkit (cuda.h, cuda.lib)
 *   - NVENC SDK (nvEncodeAPI.h, nvencodeapi.lib)
 *     https://developer.nvidia.com/nvidia-video-codec-sdk
 *
 * 二进制协议 (stdin):
 *   ┌──────────────────────────────────────────────┐
 *   │ HEADER:                                       │
 *   │   uint32 magic         = 0x4E56454E ("NVEN") │
 *   │   uint32 version       = 1                    │
 *   │   uint32 width                                │
 *   │   uint32 height                               │
 *   │   uint32 fps                                  │
 *   │   uint32 bitrate_kbps                         │
 *   │   uint32 output_path_len                      │
 *   │   char[]  output_path (UTF-8)                 │
 *   ├──────────────────────────────────────────────┤
 *   │ FRAME (重复到收到结束标记):                    │
 *   │   uint32 repeat_count   (0 = 结束)            │
 *   │   uint32 data_size      (0 = 复用上一帧)      │
 *   │   if data_size > 0:                           │
 *   │     uint8[] pixel_data  (RGBA, 未压缩)        │
 *   └──────────────────────────────────────────────┘
 *
 * 帧重复优化:
 *   - data_size=0 时，复用上一帧的 GPU 缓冲区
 *   - NVENC 会将重复帧编码为 P-skip，几乎零成本
 *   - 500 帧真正编码 + 26500 帧 skip ≈ 1.5s
 */

#define NOMINMAX
#include <windows.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <chrono>
#include <string>
#include <queue>
#include <mutex>
#include <thread>

// NVENC SDK 头文件
#include "nvEncodeAPI.h"

// ============================================================
// 二进制协议常量
// ============================================================
constexpr uint32_t PROTOCOL_MAGIC   = 0x4E56454E;  // "NVEN"
constexpr uint32_t PROTOCOL_VERSION = 1;
constexpr uint32_t PACKED_SIZE      = 4;  // 4 字节对齐

// ============================================================
// NVENC 动态库加载
// ============================================================
class NvencLibrary {
public:
    HMODULE module = nullptr;

    // NVENC API 函数指针
    NV_ENCODE_API_FUNCTION_LIST api = {};

    bool load() {
        // 尝试加载 NVENC 动态库
        module = LoadLibraryW(L"nvEncodeAPI64.dll");
        if (!module) {
            module = LoadLibraryW(L"nvEncodeAPI.dll");
        }
        if (!module) {
            std::cerr << "[ERROR] 无法加载 nvEncodeAPI64.dll，请安装 NVIDIA 驱动" << std::endl;
            return false;
        }

        // 获取 NVENC API 实例
        using NvEncodeAPICreateInstanceFunc = NVENCSTATUS(NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*);
        auto createInstance = (NvEncodeAPICreateInstanceFunc)
            GetProcAddress(module, "NvEncodeAPICreateInstance");

        if (!createInstance) {
            std::cerr << "[ERROR] 无法获取 NvEncodeAPICreateInstance" << std::endl;
            return false;
        }

        // 初始化 API 函数列表
        api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        NVENCSTATUS status = createInstance(&api);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] NvEncodeAPICreateInstance 失败: " << status << std::endl;
            return false;
        }
        return true;
    }

    ~NvencLibrary() {
        if (module) FreeLibrary(module);
    }
};

// ============================================================
// CUDA 设备管理
// ============================================================
class CudaDevice {
public:
    CUdevice  device = 0;
    CUcontext context = nullptr;

    bool init() {
        CUresult err = cuInit(0);
        if (err != CUDA_SUCCESS) {
            std::cerr << "[ERROR] cuInit 失败: " << err << std::endl;
            return false;
        }

        int deviceCount = 0;
        cuDeviceGetCount(&deviceCount);
        if (deviceCount == 0) {
            std::cerr << "[ERROR] 未找到 CUDA 设备" << std::endl;
            return false;
        }

        // 使用第一张显卡
        cuDeviceGet(&device, 0);
        char name[128];
        cuDeviceGetName(name, sizeof(name), device);
        std::cerr << "[INFO] CUDA 设备: " << name << std::endl;

        err = cuCtxCreate(&context, 0, device);
        if (err != CUDA_SUCCESS) {
            std::cerr << "[ERROR] cuCtxCreate 失败: " << err << std::endl;
            return false;
        }
        return true;
    }

    ~CudaDevice() {
        if (context) cuCtxDestroy(context);
    }
};

// ============================================================
// 帧数据
// ============================================================
struct FramePacket {
    uint32_t repeat_count;  // 输出帧重复次数
    uint32_t data_size;     // 像素数据大小 (0 = 复用上一帧)
    std::vector<uint8_t> pixels;
};

// ============================================================
// NVENC 编码器
// ============================================================
class NvencEncoder {
public:
    NvencLibrary*         lib = nullptr;
    void*                 encoder = nullptr;
    NV_ENC_INPUT_PTR      input_buffer = nullptr;
    NV_ENC_OUTPUT_PTR     bitstream_buffer = nullptr;
    uint32_t              input_buffer_pitch = 0;

    // 编码参数
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
    uint32_t bitrate_kbps = 0;
    uint64_t frame_count = 0;

    // 输出文件
    std::ofstream output_file;
    std::string   output_path;

    // 上一帧的 GPU 缓冲区 (用于帧重复优化)
    CUdeviceptr    last_frame_gpu = 0;
    uint32_t       last_frame_size = 0;

    bool init(NvencLibrary* nvenc_lib, CudaDevice* cuda_dev,
              uint32_t w, uint32_t h, uint32_t f, uint32_t br) {
        lib = nvenc_lib;
        width = w;
        height = h;
        fps = f;
        bitrate_kbps = br;

        // ---- 1. 创建 NVENC 编码器 ----
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_params = {};
        open_params.version  = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        open_params.device   = cuda_dev->context;
        open_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        open_params.apiVersion = NVENCAPI_VERSION;

        NVENCSTATUS status = lib->api.nvEncOpenEncodeSessionEx(&open_params, &encoder);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] nvEncOpenEncodeSessionEx 失败: " << status << std::endl;
            return false;
        }

        // ---- 2. 初始化编码器参数 ----
        NV_ENC_INITIALIZE_PARAMS init_params = {};
        init_params.version            = NV_ENC_INITIALIZE_PARAMS_VER;
        init_params.encodeGUID         = NV_ENC_CODEC_H264_GUID;
        init_params.encodeWidth        = width;
        init_params.encodeHeight       = height;
        init_params.darWidth           = width;
        init_params.darHeight          = height;
        init_params.frameRateNum       = fps;
        init_params.frameRateDen       = 1;
        init_params.enableEncodeAsync  = 0;
        init_params.enablePTD          = 1;  // 逐帧编码模式

        // 配置预设参数
        NV_ENC_PRESET_CONFIG preset_config = {};
        preset_config.version = NV_ENC_PRESET_CONFIG_VER;
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

        status = lib->api.nvEncGetEncodePresetConfig(encoder,
            NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P7_GUID, &preset_config);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[WARN] 使用默认预设配置" << std::endl;
            preset_config.presetCfg.rcParams.averageBitRate = bitrate_kbps * 1000;
            preset_config.presetCfg.rcParams.maxBitRate     = bitrate_kbps * 1000 * 2;
            preset_config.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        }

        // 应用码率设置
        preset_config.presetCfg.rcParams.averageBitRate = bitrate_kbps * 1000;
        preset_config.presetCfg.rcParams.maxBitRate     = bitrate_kbps * 1000 * 2;
        preset_config.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

        init_params.encodeConfig = &preset_config.presetCfg;

        status = lib->api.nvEncInitializeEncoder(encoder, &init_params);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] nvEncInitializeEncoder 失败: " << status << std::endl;
            return false;
        }

        // ---- 3. 创建输入缓冲区 ----
        NV_ENC_CREATE_INPUT_BUFFER input_buf_params = {};
        input_buf_params.version    = NV_ENC_CREATE_INPUT_BUFFER_VER;
        input_buf_params.width      = width;
        input_buf_params.height     = height;
        input_buf_params.memoryHeap = NV_ENC_INPUT_RESOURCE_TYPE_SYSTEM;
        input_buf_params.bufferFmt  = NV_ENC_BUFFER_FORMAT_ABGR;  // RGBA

        status = lib->api.nvEncCreateInputBuffer(encoder, &input_buf_params);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] nvEncCreateInputBuffer 失败: " << status << std::endl;
            return false;
        }
        input_buffer = input_buf_params.inputBuffer;
        input_buffer_pitch = input_buf_params.byteSizePerPixel * width;

        // ---- 4. 创建输出比特流缓冲区 ----
        NV_ENC_CREATE_BITSTREAM_BUFFER bitstream_params = {};
        bitstream_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        bitstream_params.size    = 2 * 1024 * 1024;  // 2MB 缓冲区

        status = lib->api.nvEncCreateBitstreamBuffer(encoder, &bitstream_params);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] nvEncCreateBitstreamBuffer 失败: " << status << std::endl;
            return false;
        }
        bitstream_buffer = bitstream_params.bitstreamBuffer;

        std::cerr << "[INFO] 编码器初始化成功: "
                  << width << "x" << height << " "
                  << fps << "fps " << bitrate_kbps << "kbps" << std::endl;
        return true;
    }

    bool open_output(const std::string& path) {
        output_path = path;
        output_file.open(path, std::ios::binary);
        if (!output_file.is_open()) {
            std::cerr << "[ERROR] 无法打开输出文件: " << path << std::endl;
            return false;
        }
        // 写入 H.264 起始码 (AVC 格式)
        output_file.write("\x00\x00\x00\x01", 4);
        return true;
    }

    /**
     * 编码一帧。
     * @param rgba_data  RGBA 像素数据 (nullptr = 复用上一帧)
     * @param repeat     这帧在输出中重复的帧数（用于进度统计）
     * @return true 成功
     */
    bool encode_frame(const uint8_t* rgba_data, uint32_t repeat) {
        if (rgba_data != nullptr) {
            // ---- 新帧: 锁定输入缓冲区，拷贝像素数据 ----
            NV_ENC_LOCK_INPUT_BUFFER lock_buf = {};
            lock_buf.version  = NV_ENC_LOCK_INPUT_BUFFER_VER;
            lock_buf.inputBuffer = input_buffer;

            NVENCSTATUS status = lib->api.nvEncLockInputBuffer(encoder, &lock_buf);
            if (status != NV_ENC_SUCCESS) {
                std::cerr << "[ERROR] nvEncLockInputBuffer 失败: " << status << std::endl;
                return false;
            }

            // 拷贝 RGBA 数据到输入缓冲区
            uint8_t* dst = (uint8_t*)lock_buf.bufferDataPlane[0];
            size_t src_pitch = width * 4;
            size_t dst_pitch = lock_buf.pitch[0];
            for (uint32_t y = 0; y < height; y++) {
                memcpy(dst + y * dst_pitch, rgba_data + y * src_pitch, src_pitch);
            }

            lib->api.nvEncUnlockInputBuffer(encoder, input_buffer);
        }
        // 如果是重复帧，input_buffer 中已经是上一帧的数据，直接编码

        // ---- 编码 ----
        NV_ENC_PIC_PARAMS pic_params = {};
        pic_params.version         = NV_ENC_PIC_PARAMS_VER;
        pic_params.inputBuffer     = input_buffer;
        pic_params.bufferFmt       = NV_ENC_BUFFER_FORMAT_ABGR;
        pic_params.inputWidth      = width;
        pic_params.inputHeight     = height;
        pic_params.inputPitch      = input_buffer_pitch;
        pic_params.outputBitstream = bitstream_buffer;
        pic_params.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;

        // 第一帧为 IDR 帧，后续为 P 帧
        if (frame_count == 0) {
            pic_params.encodePicFlags = NV_ENC_PIC_FLAG_FORCEINTRA |
                                        NV_ENC_PIC_FLAG_OUTPUT_DATA_SET;
            pic_params.picType = NV_ENC_PIC_TYPE_IDR;
        } else {
            pic_params.encodePicFlags = NV_ENC_PIC_FLAG_OUTPUT_DATA_SET;
            pic_params.picType = NV_ENC_PIC_TYPE_P;
        }

        // 重复帧优化: 设置 refPic=0 避免用作参考帧
        if (rgba_data == nullptr) {
            pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_EOS;
        }

        NVENCSTATUS status = lib->api.nvEncEncodePicture(encoder, &pic_params);
        if (status != NV_ENC_SUCCESS) {
            std::cerr << "[ERROR] nvEncEncodePicture 失败: " << status << std::endl;
            return false;
        }

        // ---- 读取编码后的比特流 ----
        NV_ENC_LOCK_BITSTREAM lock_bitstream = {};
        lock_bitstream.version        = NV_ENC_LOCK_BITSTREAM_VER;
        lock_bitstream.outputBitstream = bitstream_buffer;
        lock_bitstream.doNotWait      = 0;

        status = lib->api.nvEncLockBitstream(encoder, &lock_bitstream);
        if (status != NV_ENC_SUCCESS) {
            // 没有输出数据（可能缓存了），正常
            return true;
        }

        // 写入 H.264 比特流
        if (lock_bitstream.bitstreamSizeInBytes > 0) {
            // 跳过起始码 (NVENC 输出可能包含起始码)
            const uint8_t* data = (const uint8_t*)lock_bitstream.bitstreamBuffer;
            uint32_t size = lock_bitstream.bitstreamSizeInBytes;

            // 检查是否已有起始码
            if (size > 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
                output_file.write((const char*)data, size);
            } else {
                output_file.write("\x00\x00\x00\x01", 4);
                output_file.write((const char*)data, size);
            }
        }

        lib->api.nvEncUnlockBitstream(encoder, bitstream_buffer);
        frame_count++;
        return true;
    }

    void flush() {
        // 发送结束标记，让编码器输出所有缓存帧
        NV_ENC_PIC_PARAMS pic_params = {};
        pic_params.version         = NV_ENC_PIC_PARAMS_VER;
        pic_params.encodePicFlags  = NV_ENC_PIC_FLAG_EOS;
        pic_params.inputBuffer     = input_buffer;
        pic_params.bufferFmt       = NV_ENC_BUFFER_FORMAT_ABGR;
        pic_params.outputBitstream = bitstream_buffer;

        lib->api.nvEncEncodePicture(encoder, &pic_params);
    }

    void close() {
        if (output_file.is_open()) {
            output_file.close();
        }
        if (bitstream_buffer) {
            lib->api.nvEncDestroyBitstreamBuffer(encoder, bitstream_buffer);
            bitstream_buffer = nullptr;
        }
        if (input_buffer) {
            lib->api.nvEncDestroyInputBuffer(encoder, input_buffer);
            input_buffer = nullptr;
        }
        if (encoder) {
            lib->api.nvEncDestroyEncoder(encoder);
            encoder = nullptr;
        }
    }

    ~NvencEncoder() {
        close();
    }
};

// ============================================================
// H.264 → MP4 封装 (使用 FFmpeg)
// ============================================================
bool mux_to_mp4(const std::string& h264_path, const std::string& mp4_path) {
    std::string cmd = "ffmpeg -y -f h264 -i \"" + h264_path +
                      "\" -c copy \"" + mp4_path + "\" 2>nul";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[WARN] FFmpeg mux 失败, H.264 文件保留: " << h264_path << std::endl;
        return false;
    }
    // 删除临时 H.264 文件
    std::remove(h264_path.c_str());
    return true;
}

// ============================================================
// 主函数
// ============================================================
int main() {
    // ---- 读取 stdin 二进制头部 ----
    auto read_exact = [](uint8_t* buf, size_t size) -> bool {
        size_t offset = 0;
        while (offset < size) {
            size_t n = fread(buf + offset, 1, size - offset, stdin);
            if (n == 0) return false;
            offset += n;
        }
        return true;
    };

    auto read_u32 = [&]() -> uint32_t {
        uint32_t val;
        if (!read_exact((uint8_t*)&val, 4)) return 0;
        return val;
    };

    // ---- 读取头部 ----
    uint32_t magic = read_u32();
    if (magic != PROTOCOL_MAGIC) {
        std::cerr << "[ERROR] 无效的协议头部: magic=0x"
                  << std::hex << magic << std::dec << std::endl;
        return 1;
    }

    uint32_t version    = read_u32();
    uint32_t width      = read_u32();
    uint32_t height     = read_u32();
    uint32_t fps        = read_u32();
    uint32_t bitrate    = read_u32();
    uint32_t path_len   = read_u32();

    if (version != PROTOCOL_VERSION) {
        std::cerr << "[ERROR] 协议版本不匹配: " << version << std::endl;
        return 1;
    }

    std::string output_path(path_len, '\0');
    if (!read_exact((uint8_t*)output_path.data(), path_len)) {
        std::cerr << "[ERROR] 读取输出路径失败" << std::endl;
        return 1;
    }

    std::cerr << "[INFO] 收到编码请求: "
              << width << "x" << height << " "
              << fps << "fps "
              << bitrate << "kbps"
              << " → " << output_path << std::endl;

    // ---- 初始化 CUDA ----
    CudaDevice cuda;
    if (!cuda.init()) {
        return 1;
    }

    // ---- 加载 NVENC ----
    NvencLibrary nvenc_lib;
    if (!nvenc_lib.load()) {
        return 1;
    }

    // ---- 初始化编码器 ----
    NvencEncoder encoder;
    if (!encoder.init(&nvenc_lib, &cuda, width, height, fps, bitrate)) {
        return 1;
    }

    // 输出到临时 H.264 文件
    std::string h264_path = output_path + ".h264.tmp";
    if (!encoder.open_output(h264_path)) {
        return 1;
    }

    // ---- 帧编码循环 ----
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t total_output_frames = 0;
    uint64_t unique_frames = 0;
    uint64_t repeat_frames = 0;
    size_t frame_bytes = (size_t)width * height * 4;

    while (true) {
        uint32_t repeat_count = read_u32();
        uint32_t data_size    = read_u32();

        // 结束标记
        if (repeat_count == 0 && data_size == 0) {
            break;
        }

        if (data_size > 0) {
            // ---- 新帧: 读取像素数据 ----
            if (data_size != frame_bytes) {
                std::cerr << "[ERROR] 帧数据大小不匹配: 期望 "
                          << frame_bytes << ", 收到 " << data_size << std::endl;
                return 1;
            }

            std::vector<uint8_t> pixels(data_size);
            if (!read_exact(pixels.data(), data_size)) {
                std::cerr << "[ERROR] 读取帧数据失败" << std::endl;
                return 1;
            }

            // 编码这一帧（首次）
            if (!encoder.encode_frame(pixels.data(), repeat_count)) {
                return 1;
            }
            unique_frames++;
            total_output_frames++;

            // 重复帧: 复用同一像素数据，但作为独立帧编码
            // 注意: 对于重复帧，NVENC 会自动做 P-skip
            for (uint32_t r = 1; r < repeat_count; r++) {
                if (!encoder.encode_frame(pixels.data(), repeat_count)) {
                    return 1;
                }
                repeat_frames++;
                total_output_frames++;
            }
        } else {
            // ---- 复用上一帧 (data_size == 0) ----
            // 重复使用上一帧的像素数据
            for (uint32_t r = 0; r < repeat_count; r++) {
                if (!encoder.encode_frame(nullptr, repeat_count)) {
                    return 1;
                }
                repeat_frames++;
                total_output_frames++;
            }
        }

        // 进度报告
        if (total_output_frames % 1000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cerr << "[INFO] 已编码 " << total_output_frames << " 帧, "
                      << "耗时 " << elapsed << "s, "
                      << "速度 " << (total_output_frames / elapsed) << " fps"
                      << std::endl;
        }
    }

    // ---- 刷出缓存帧 ----
    encoder.flush();

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::cerr << "\n[INFO] 编码完成!" << std::endl;
    std::cerr << "  输出帧数:   " << total_output_frames << std::endl;
    std::cerr << "  唯一帧数:   " << unique_frames << std::endl;
    std::cerr << "  重复帧数:   " << repeat_frames << std::endl;
    std::cerr << "  编码耗时:   " << elapsed << "s" << std::endl;
    std::cerr << "  编码速度:   " << (total_output_frames / elapsed) << " fps" << std::endl;

    // ---- 关闭编码器 ----
    encoder.close();

    // ---- Mux 到 MP4 ----
    std::cerr << "[INFO] 封装 MP4..." << std::endl;
    if (mux_to_mp4(h264_path, output_path)) {
        std::cerr << "[INFO] 输出: " << output_path << std::endl;
    }

    // 输出结果到 stdout (Python 读取)
    std::cout << total_output_frames << std::endl;

    return 0;
}