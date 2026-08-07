@echo off
REM ============================================================
REM 编译 nvenc_encoder.exe
REM 需要: Visual Studio 2022 + CUDA Toolkit + NVENC SDK
REM ============================================================

echo [INFO] 编译 nvenc_encoder.cpp...

REM 设置 VS 环境 (根据你的安装路径调整)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 设置 CUDA 路径 (根据你的安装路径调整)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set NVENC_SDK_PATH=C:\Video_Codec_SDK_12.2

cl /EHsc /std:c++17 /O2 /MD /I"%CUDA_PATH%\include" /I"%NVENC_SDK_PATH%\Interface" ^
    nvenc_encoder.cpp ^
    /link /LIBPATH:"%CUDA_PATH%\lib\x64" /LIBPATH:"%NVENC_SDK_PATH%\Lib\x64" ^
    cuda.lib nvencodeapi.lib

if %ERRORLEVEL% == 0 (
    echo [OK] 编译成功: nvenc_encoder.exe
) else (
    echo [ERROR] 编译失败
)