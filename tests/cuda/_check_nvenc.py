"""检查 NVENC API 可用性。"""
import ctypes
import os

# Load the DLL
dll = ctypes.cdll.LoadLibrary('nvEncodeAPI64.dll')

# Check if NvEncodeAPICreateInstance is available
try:
    func = dll.NvEncodeAPICreateInstance
    print(f"NvEncodeAPICreateInstance found at: {func}")
except AttributeError as e:
    print(f"NvEncodeAPICreateInstance NOT found: {e}")
    # Try alternate names
    for name in dir(dll):
        if 'nvenc' in name.lower() or 'nv' in name.lower()[:2]:
            print(f"  Found: {name}")

# List all NvEnc* exports
print("\nAll NvEnc* exports:")
for name in sorted(dir(dll)):
    if name.startswith('NvEnc'):
        print(f"  {name}")

# Check DLL version info
dll_path = 'C:\\Windows\\System32\\nvEncodeAPI64.dll'
if os.path.exists(dll_path):
    print(f"\nDLL: {dll_path}")
    print(f"Size: {os.path.getsize(dll_path)} bytes")
    
    # Get file version info
    try:
        import win32api
        info = win32api.GetFileVersionInfo(dll_path, '\\')
        print(f"Version: {info.get('FileVersionMS', '?')}.{info.get('FileVersionLS', '?')}")
    except ImportError:
        # Try using struct
        try:
            import subprocess
            result = subprocess.run(['powershell', 
                '(Get-Item "C:\\Windows\\System32\\nvEncodeAPI64.dll").VersionInfo | Format-List'],
                capture_output=True, text=True, timeout=10)
            print(result.stdout)
        except Exception as e:
            print(f"Could not get version: {e}")