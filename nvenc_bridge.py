"""
NVENC 硬件编码桥接模块
========================
用法:
    1. 让朋友把 nvenc_encoder.cpp 编译成 nvenc_encoder.exe
    2. 把 exe 放在脚本同目录下
    3. Python 调用:

        from nvenc_bridge import NvencEncoder

        enc = NvencEncoder("nvenc_encoder.exe")
        enc.start(width=1920, height=1080, fps=60, bitrate_kbps=10000,
                  output_path="result.mp4")

        for rgba_bytes, repeat_count in frames:
            enc.submit_frame(rgba_bytes, repeat_count)

        enc.finish()
        print(f"编码完成, 耗时 {enc.elapsed:.1f}s")

帧重复优化:
    - 如果一帧和上一帧视觉相同（比如同一个音符持续期间）
    - 传入 repeat_count > 1，编码器会用 P-skip 跳过这些帧
    - 每帧 skip 仅 ~0.01ms，远小于正常编码 ~2.5ms
"""

import struct
import subprocess
import time
import os
import sys
import threading


class NvencEncoder:
    """NVENC 硬件编码器 (Python -> C++ 桥接)"""

    # 二进制协议常量
    MAGIC = 0x4E56454E  # "NVEN"
    VERSION = 1

    def __init__(self, exe_path: str = "nvenc_encoder.exe"):
        self.exe_path = exe_path
        self.process: subprocess.Popen | None = None
        self.elapsed = 0.0
        self._frame_count = 0
        self._stderr_thread = None

    def start(self, width: int, height: int, fps: int,
              bitrate_kbps: int, output_path: str):
        """启动编码器进程，发送头部信息。"""
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(
                f"找不到 nvenc_encoder.exe，请先编译: {self.exe_path}"
            )

        self.process = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 启动 stderr 读取线程（避免 pipe 卡死）
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        # 编码头部
        out_path_bytes = output_path.encode("utf-8")
        header = struct.pack(
            "<IIIIII",  # 小端序
            self.MAGIC,
            self.VERSION,
            width,
            height,
            fps,
            bitrate_kbps,
        )
        header += struct.pack("<I", len(out_path_bytes))
        header += out_path_bytes

        self._write_all(header)
        self._start_time = time.time()

    def submit_frame(self, rgba_bytes: bytes | None, repeat_count: int):
        """提交一帧。

        Args:
            rgba_bytes: RGBA 像素数据 (width*height*4 字节)。
                        如果是 None，表示复用上一帧的像素数据（更高效）。
            repeat_count: 这帧在输出中重复多少次（帧重复优化）。
                          例如一个音符持续 30 帧，传 30。
        """
        if rgba_bytes is not None:
            data_size = len(rgba_bytes)
            packet = struct.pack("<II", repeat_count, data_size) + rgba_bytes
        else:
            packet = struct.pack("<II", repeat_count, 0)

        self._write_all(packet)
        self._frame_count += repeat_count

    def finish(self) -> int:
        """结束编码。返回输出文件的总帧数。"""
        if self.process is None:
            raise RuntimeError("编码器未启动")

        # 发送结束标记 (repeat_count=0, data_size=0)
        self._write_all(struct.pack("<II", 0, 0))
        self.process.stdin.close()
        self.process.wait()

        self.elapsed = time.time() - self._start_time
        return self._frame_count

    def _write_all(self, data: bytes):
        if self.process and self.process.stdin:
            self.process.stdin.write(data)
            self.process.stdin.flush()

    def _read_stderr(self):
        """读取编码器的 stderr 日志，打印到控制台。"""
        if self.process and self.process.stderr:
            for line in iter(self.process.stderr.readline, b""):
                sys.stderr.write(f"[NVENC] {line.decode('utf-8', errors='replace')}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.returncode is None:
            self.finish()


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 模拟：500 张唯一帧，每帧重复 36 次 = 18000 帧输出 (60fps × 5min)
    WIDTH, HEIGHT = 1920, 1080
    FPS = 60
    UNIQUE_FRAMES = 500
    AVG_REPEAT = 36  # 平均每帧重复 36 次

    # 生成模拟帧数据（纯色填充）
    print(f"生成 {UNIQUE_FRAMES} 张模拟帧...")
    fake_frames = []
    for i in range(UNIQUE_FRAMES):
        r = (i * 37) % 256
        g = (i * 73) % 256
        b = (i * 151) % 256
        frame = bytes([r, g, b, 255]) * (WIDTH * HEIGHT)
        fake_frames.append(frame)

    print(f"开始编码: {WIDTH}x{HEIGHT} {FPS}fps, "
          f"{UNIQUE_FRAMES} 唯一帧 → {UNIQUE_FRAMES * AVG_REPEAT} 输出帧")

    enc = NvencEncoder("nvenc_encoder.exe")
    try:
        enc.start(WIDTH, HEIGHT, FPS, bitrate_kbps=10000,
                  output_path="test_output.mp4")

        for i, frame in enumerate(fake_frames):
            enc.submit_frame(frame, repeat_count=AVG_REPEAT)
            if (i + 1) % 50 == 0:
                print(f"  已提交 {i+1}/{UNIQUE_FRAMES} 唯一帧")

        total_frames = enc.finish()
        print(f"编码完成: {total_frames} 帧, 耗时 {enc.elapsed:.1f}s")
        print(f"编码速度: {total_frames / enc.elapsed:.0f} fps")

    except FileNotFoundError as e:
        print(f"请先编译 nvenc_encoder.exe:\n  {e}")