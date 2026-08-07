<div align="center">

<img src="icon.png" height="90" width="90"/>

# ustxPlayer-preview

`v26h07` · 基于 [ustxPlayer](https://github.com/SYEternalR/ustxPlayer) 二次开发的 USTX 工程可视化工具。

![GitHub Release](https://img.shields.io/github/v/release/lyrinXD/ustxPlayer?style=for-the-badge)
![GitHub All Releases](https://img.shields.io/github/downloads/lyrinXD/ustxPlayer/total?style=for-the-badge)
![GitHub Stars](https://img.shields.io/github/stars/lyrinXD/ustxPlayer?style=for-the-badge)

[配布视频](https://www.bilibili.com/video/BV1YjcwzVEcX "bilibili弹幕网") | [更新日志](UPDATELOG.md)

![软件截图](https://github.com/user-attachments/assets/81190099-3f1d-4700-ac68-9588f30c04cb)

</div>

> [!NOTE]
> 本项目是 ustxPlayer 的二次开发版本，核心定位从 UST 转向 **USTX**，并围绕播放体验、样式系统、歌词支持等方面进行了大量改进。

## 主要特性

### USTX 工程支持
- 原生支持 OpenUtau 的 `.ustx` 工程文件（基于 YAML 解析），**无需手动尝试编码**。
- **多音轨选择**：USTX 文件含多条可解析音轨时，加载前自动弹出音轨选择窗口，可按名称与音符数挑选需要加载的音轨。
- 旧版 `.ust` 用户可通过 [UtaFormatix](https://utaformatix.tk/) 等工具转换后使用。

### 逐字样式系统
- 在项目中为**每一个字**精确定义样式（字体、颜色、位置等）。
- 配套直观的样式编辑界面，支持批量编辑。

### 增强播放体验
- 支持**导入音频与ustx文件同步播放**，更直观的同时方便在剪辑软件中精确对轨。
- 播放控制：暂停 / 快进 / 快退 / 音量调节 / 倍速播放（快捷键详见软件内说明）。

### 导出 MP4 视频
- 点击播放按钮后选择「**导出为视频**」，进入内嵌于主窗口的**渲染导出页**，将 USTX 可视化画面（含逐字歌词、音高线、LRC 多语言歌词等）**离屏渲染**并编码为视频，画面与播放器所见一致。
- **全参数设置**：分辨率（720P / 1080P / 2K / 4K / 自定义）、帧率（30 / 60 / 90 / 120 / 自定义，默认 60）。
- **GPU 硬件加速渲染**：支持 CUDA / OpenGL / CPU 三种渲染后端，自动检测硬件可用性并禁用不可用项；CUDA 渲染可并行多 stream 渲染唯一帧后交给 FFmpeg 帧重复编码。
- **智能预估**：自动计算唯一帧数、渲染并发与编码并发，预估导出耗时。
- **渲染模式**：支持「渲染完再编码 / 边渲染边编码 / 自动」三种模式。
- **进度可视化**：进度条 + 阶段文字 + Windows 任务栏进度，导出中可取消。
- **错误日志**：失败时显示原因并可直接打开日志文件，便于提交 [Issues](https://github.com/lyrinXD/ustxPlayer/issues) 反馈。
- **内置 ffmpeg**：通过依赖 [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) 自带编码器，无需用户额外安装。

> [!WARNING]
> 导出功能为**试验性功能**，画面细节可能与播放器略有差异；如遇错误请在 [Issues](https://github.com/lyrinXD/ustxPlayer/issues) 反馈并附上导出的错误日志。

### 导出计时
- 导出时自动记录每个阶段（预计算 / 渲染 / 编码）的耗时，汇总输出到日志，包括各阶段耗时占比、实际帧率等，便于性能分析和调优。

### 启动动画
- 软件启动时展示现代化圆角卡片动画（圆形应用图标、应用名称、版本号与加载进度条），跟随系统亮/暗主题，主窗口就绪后平滑淡出。

### 多语言 LRC 歌词
- 支持 `.lrc` 文件的**交错**与**独立**多语言格式。
- 理论支持任意行数，推荐 1~3 行以获得最佳显示效果。

### 更多自定义
- 支持自定义界面**强调色**，适配深色模式。
- 可修改歌词/信息字体、信息颜色等。
- 可隐藏软件版权信息。


## 安装与运行

### 环境要求
- Windows 10/11
- Python 3.10+
- 导出 MP4 使用依赖自带的 ffmpeg（imageio-ffmpeg），无需额外安装
- CUDA 硬件加速渲染需要 NVIDIA 显卡 + CUDA 驱动（可选，无显卡时自动回退 OpenGL / CPU）

### 从源码运行

```bash
# 1. 克隆仓库
git clone https://github.com/github-2000-07-05/ustxPlayer-preview.git
cd ustxPlayer-preview

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
python main.py
```

### 打包为可执行文件

```bash
# 确保已安装 Nuitka（推荐）或 PyInstaller
pip install -r requirements.txt

# 方式一：使用 build.bat（推荐，Nuitka 会自动处理编译器，产出单文件）
build.bat

# 方式二：手动执行 Nuitka 单文件模式（需要自行准备 C 编译器）
pip install "Nuitka[all]"
python -m nuitka --onefile --enable-plugin=pyside6 main.py

# 方式三：使用 PyInstaller 单文件模式
pip install pyinstaller
python -m PyInstaller --onefile --windowed --icon=icon.ico --name=ustxPlayer \
    --add-data="icon.ico;." --add-data="Terms.txt;." --add-data="LICENSE;." \
    --collect-all qfluentwidgets --collect-all qframelesswindow main.py
```

> 打包完成后可执行文件位于 `dist\ustxPlayer.exe`（单文件），首次编译预计耗时 10-20 分钟，后续编译会被缓存加速。

## 使用提示

- 歌词推荐使用 **交错** 或 **独立** 格式；合并格式可能显示异常。
- 工程文件（`.uplr`）采用全新格式，**不兼容旧版 ustxPlayer 的 `.uplr` 文件**。新格式内嵌 USTX 内容，可独立分发，无需额外携带 `.ustx` 文件。
- 工程文件（.uplr）中的 ustx 等文件路径**可以为空**，故工程文件可做模板使用。
- 同时打开两个界面会出现异常的BUG已在v26h04版本修复修复。


## 致谢

本项目时基于 **[ustxPlayer](https://github.com/SYEternalR/ustxPlayer)** 二次开发的第三方优化版，由github-2000-07-05修改，原项目由 **[SYEternal_R](https://github.com/SYEternalR)** 与 **[灰棱HiRenG](https://github.com/HiRenG1145)** 创建。

### 使用的资源与库

- [PySide6](https://www.qt.io/)
- [PySide6-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets/tree/PySide6)
- [loguru](https://github.com/Delgan/loguru)
- [PyYAML](https://github.com/yaml/pyyaml)
- [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/)（内置 ffmpeg 编码器）
- [cupy](https://cupy.dev/)（CUDA 硬件加速渲染，可选）

## 协议与许可

本项目沿用原项目（ustxPlayer）的使用协议，使用前请务必阅读并同意相关使用协议：

- 软件内入口：`其他 > 关于软件`，点击「上游使用协议」或「GNU LGPL v3.0」即可在软件内直接查看全文
- 程序目录下 [`Terms.txt`](Terms.txt)（上游使用协议）与 [`LICENSE`](LICENSE)（GNU LGPL v3.0）

ustxPlayer 原项目版权由 SYEternalR 所有。本项目（ustxPlayer-preview）在 ustxPlayer 基础上进行二次开发，授权给符合条件的用户免费使用。

本工具在开发过程中使用了 AI 工具进行辅助开发。

---

感谢使用，玩得开心！
