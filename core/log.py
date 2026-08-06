# log.py — 日志配置
"""基于 loguru 的全局日志。

每次启动创建一个带时间戳的独立日志文件，最多保留 50 个，
超出时自动删除最旧的。

用法:
    from core.log import logger

    logger.info("正常信息")
    logger.debug("调试信息")
    logger.exception("自动附完整堆栈")
"""

import sys
import os
import glob
from datetime import datetime

from loguru import logger

logger.remove()

# 统一使用程序根目录（与 settings_manager.program_root 一致，基于 sys.argv[0]），
# 日志存放在根目录下的 log/ 子目录中。
_log_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "log")

# 确保日志目录存在
try:
    os.makedirs(_log_dir, exist_ok=True)
except OSError:
    pass

# 日志文件名前缀 + 最大保留数量
_LOG_PREFIX = "ustxPlayer-preview"
_MAX_LOG_FILES = 50

# 本次会话的日志文件路径
_current_log_path = os.path.join(
    _log_dir,
    f"{_LOG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


def _cleanup_old_logs() -> None:
    """删除超出 _MAX_LOG_FILES 数量的最旧日志文件。"""
    pattern = os.path.join(_log_dir, f"{_LOG_PREFIX}_*.log")
    files = glob.glob(pattern)
    if len(files) <= _MAX_LOG_FILES:
        return
    # 按修改时间升序排列（最旧在前）
    files.sort(key=lambda f: os.path.getmtime(f))
    excess = len(files) - _MAX_LOG_FILES
    for f in files[:excess]:
        try:
            os.remove(f)
        except OSError:
            pass


# 启动时清理旧日志
try:
    _cleanup_old_logs()
except Exception:
    pass

try:
    logger.add(
        _current_log_path,
        level="DEBUG",
        rotation="1 MB",
        retention=_MAX_LOG_FILES,
    )
except Exception:
    # 日志文件不可写时降级（如目录无权限），不应阻塞主程序启动
    pass

if sys.stdout is not None:
    logger.add(sys.stdout, level="INFO", colorize=True)


def get_log_file_path() -> str:
    """返回本次会话的日志文件完整路径（渲染失败时展示给用户）。"""
    return _current_log_path
