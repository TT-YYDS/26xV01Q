"""Sizector S 2.776 C API acquisition, with explicit ownership of native frames."""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import re
import threading
import time
from tempfile import NamedTemporaryFile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK = ROOT / "盛相SDK"
RUNTIME_DIR = ROOT / "settings" / "sizector_runtime" / "x64"
WORKING_MODES = {0: "快速 3D", 1: "标准 3D", 2: "精准 3D", 3: "超准 3D",
                 8: "超快 3D", 9: "快速精准 3D", 10: "超动态 3D"}
DEFAULT_VIEWS = 5
EXPOSURE_MODES = {0: "手动 / HDR", 1: "重复曝光", 2: "自动 NHDR", 3: "自动 PHDR"}
EXPOSURE_LIMITS = {"mode": (0, 3), "number": (1, 3), "gain": (0, 3),
                   "hdr_priority": (0, 7), "phdr_quality": (0, 100),
                   "intensity_1": (0.1, 100), "intensity_2": (0.1, 100),
                   "intensity_3": (0.1, 100), "gray_intensity": (0.1, 100)}
DEFAULT_EXPOSURE = {"mode": 0, "number": 1, "gain": 0, "hdr_priority": 0,
                    "phdr_quality": 50, "intensity_1": 1.0, "intensity_2": 5.0,
                    "intensity_3": 10.0, "gray_intensity": 5.0}


class CameraError(RuntimeError):
    pass


def validate_exposure(values):
    if not isinstance(values, dict) or set(values) != set(EXPOSURE_LIMITS):
        raise CameraError("曝光参数字段不完整或无法识别")
    result = {}
    for key, (low, high) in EXPOSURE_LIMITS.items():
        value = values[key]
        floating = isinstance(low, float)
        if (isinstance(value, bool) or not isinstance(value, (int, float) if floating else int)
                or not math.isfinite(value) or not low <= value <= high):
            raise CameraError(f"曝光参数 {key} 超出范围 [{low}, {high}]")
        result[key] = float(value) if floating else int(value)
    return result


class ExposureBasic(ct.Structure):
    _fields_ = [("ExposureMode", ct.c_int), ("ExposureNumber", ct.c_ubyte),
                ("AutoHDRPriority", ct.c_ubyte), ("AutoPHDRQuality", ct.c_ubyte)]


class ExposureSettings(ct.Structure):
    _fields_ = [("ExposureBasicSetting", ExposureBasic), ("UserGain", ct.c_ubyte),
                ("ExposureIntensity3D_1st", ct.c_float), ("ExposureIntensity3D_2nd", ct.c_float),
                ("ExposureIntensity3D_3rd", ct.c_float), ("ExposureIntensity2D", ct.c_float),
                ("MultiHeadExpoVarMode", ct.c_ubyte), ("MultiHeadExpoVarRatio", ct.c_ubyte)]

    def values(self):
        basic = self.ExposureBasicSetting
        return {"mode": basic.ExposureMode, "number": basic.ExposureNumber,
                "hdr_priority": basic.AutoHDRPriority, "phdr_quality": basic.AutoPHDRQuality,
                "gain": self.UserGain, "intensity_1": self.ExposureIntensity3D_1st,
                "intensity_2": self.ExposureIntensity3D_2nd, "intensity_3": self.ExposureIntensity3D_3rd,
                "gray_intensity": self.ExposureIntensity2D}

    def update_values(self, values):
        self.ExposureBasicSetting = ExposureBasic(values["mode"], values["number"],
                                                 values["hdr_priority"], values["phdr_quality"])
        self.UserGain = values["gain"]
        for key, member in (("intensity_1", "ExposureIntensity3D_1st"),
                            ("intensity_2", "ExposureIntensity3D_2nd"),
                            ("intensity_3", "ExposureIntensity3D_3rd"),
                            ("gray_intensity", "ExposureIntensity2D")):
            setattr(self, member, values[key])


def native_path(path):
    # The supplied DLL's char* file APIs use the Windows ANSI code page, not UTF-8.
    try:
        return str(Path(path).resolve()).encode("mbcs")
    except UnicodeEncodeError as error:
        raise CameraError("SDK 无法识别该路径，请使用当前 Windows 语言支持的路径") from error


class DeviceInfo(ct.Structure):
    _fields_ = [("DeviceName", ct.c_wchar * 32), ("DeviceSerialNumber", ct.c_wchar * 32),
                ("IsActivated", ct.c_bool), ("HWVersion", ct.c_float), ("FWVersion", ct.c_float),
                ("MinimumSDKVersion", ct.c_float), ("SensorWidth", ct.c_uint),
                ("SensorHeight", ct.c_uint), ("SensorResolution", ct.c_uint),
                ("BinningPossibility", ct.c_bool), ("ExternalExposurePossibility", ct.c_bool),
                ("MultiHeadConfig", ct.c_int), ("SensorBinningWidth", ct.c_uint),
                ("SensorBinningHeight", ct.c_uint), ("SensorBinningResolution", ct.c_uint),
                ("XMax", ct.c_float), ("YMax", ct.c_float), ("ZMax", ct.c_float),
                ("InterfaceType", ct.c_int)]


class TriggerInfo(ct.Structure):
    _fields_ = [("Source", ct.c_int), ("SN", ct.c_uint)]


class DataInfo(ct.Structure):
    _fields_ = [("DataFormat", ct.c_int), ("DataNumber", ct.c_ubyte), ("DataTag", ct.c_ushort),
                ("XPixResolution", ct.c_ushort), ("YPixResolution", ct.c_ushort),
                ("DeviceTimeStamp", ct.c_uint), ("PCTimeStamp", ct.c_longlong), ("TSN", TriggerInfo)]


class HeadPack(ct.Structure):
    # The supplied 2.776 headers define a 2048-byte head. Keep unused settings opaque.
    _fields_ = [("DeviceInfo", DeviceInfo), ("DataInfo", DataInfo), ("Settings", ct.c_ubyte * 1824)]


class Frame(ct.Structure):
    _fields_ = [("FrameInfo", HeadPack), ("Data", ct.POINTER(ct.c_ubyte))]


FLOAT_POINT = np.dtype([("gray", "u1"), ("mask", "u1"), ("reserved", "u1", (2,)),
                        ("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
PLY_POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def frame_points(frame):
    info = frame.FrameInfo.DataInfo
    width, height = int(info.XPixResolution), int(info.YPixResolution)
    count = width * height
    if info.DataFormat != 0:
        raise CameraError(f"需要浮点 3D 点云，实际格式为 {info.DataFormat}")
    if not frame.Data or not 0 < count <= 50_000_000:
        raise CameraError("相机帧为空或分辨率无效")
    records = np.frombuffer(ct.string_at(frame.Data, count * FLOAT_POINT.itemsize), dtype=FLOAT_POINT)
    xyz = np.column_stack([records[key] for key in ("x", "y", "z")])
    valid = ((records["mask"] & 7) == 0) & np.isfinite(xyz).all(axis=1)
    if not valid.any():
        raise CameraError("本次采集没有有效三维点")
    # Copy before the camera reuses its buffer or a loaded frame is freed.
    gray_image = records["gray"].reshape(height, width).copy()
    return xyz[valid], records["gray"][valid].copy(), gray_image


def write_capture_ply(path, xyz, gray):
    path = Path(path)
    vertices = np.empty(len(xyz), dtype=PLY_POINT)
    for index, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, index]
    for key in ("red", "green", "blue"):
        vertices[key] = gray
    header = ("ply\nformat binary_little_endian 1.0\ncomment Sizector S coordinates in mm\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".capture", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header.encode("ascii"))
            stream.write(vertices.tobytes())
        if path.exists():
            raise CameraError(f"不能覆盖已有点云：{path}")
        # Windows rename fails rather than replacing an existing destination.
        temporary.rename(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SizectorCamera:
    """Calls are serialized; DLL loading is deferred until an explicit user action."""

    def __init__(self, sdk_root=DEFAULT_SDK):
        self.sdk_root = Path(sdk_root).resolve()
        self.api = None
        self.handle = None
        self.device = None
        self.version = None
        self._lock = threading.RLock()
        self._dll_directories = []

    def _load(self):
        if self.api is not None:
            return
        if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
            raise CameraError("盛相 SDK 需要 Windows 64 位 Python")
        dll = self.sdk_root / "02_Binary" / "02_SizectorS_API" / "x64" / "MPSizectorS_API.dll"
        if not dll.is_file():
            raise CameraError(f"SDK DLL 不存在：{dll}")
        try:
            for directory in (dll.parent, RUNTIME_DIR):
                if directory.is_dir():
                    self._dll_directories.append(os.add_dll_directory(str(directory)))
            api = ct.CDLL(str(dll))
        except OSError as error:
            self._release_directories()
            raise CameraError("SDK 加载失败。需要 VC++ 2012 x64 运行库 "
                              "(MSVCR110 / MSVCP110 / VCOMP110)。详见 README 的相机接入章节。"
                              f"\n{error}") from error
        signatures = {
            "GetAPIVersion": (ct.c_float, []), "Create": (ct.c_void_p, [ct.c_int, ct.c_int]),
            "Delete": (None, [ct.c_void_p]), "UpdateDeviceList": (ct.c_bool, [ct.c_void_p]),
            "GetDeviceCount": (ct.c_int, [ct.c_void_p]),
            "GetDeviceInfo": (None, [ct.c_void_p, ct.POINTER(DeviceInfo), ct.c_int]),
            "Open": (ct.c_bool, [ct.c_void_p, DeviceInfo]), "Close": (None, [ct.c_void_p]),
            "GetDeviceState": (ct.c_int, [ct.c_void_p]),
            "SetAutoReconnect": (None, [ct.c_void_p, ct.c_bool]),
            "GetHoldState": (ct.c_bool, [ct.c_void_p, ct.POINTER(ct.c_bool)]),
            "SetHoldState": (ct.c_bool, [ct.c_void_p, ct.c_bool]),
            "GetExposureSettings": (ct.c_bool, [ct.c_void_p, ct.POINTER(ExposureSettings)]),
            "SetExposureSettings": (ct.c_bool, [ct.c_void_p, ExposureSettings]),
            "SetTriggerSource": (ct.c_bool, [ct.c_void_p, ct.c_int]),
            "SetDataOutMode": (ct.c_bool, [ct.c_void_p, ct.c_int]),
            "SetWorkingMode": (ct.c_bool, [ct.c_void_p, ct.c_int]),
            "ReadSettingsFromFile": (ct.c_bool, [ct.c_void_p, ct.c_char_p]),
            "WriteSettingsToFile": (ct.c_bool, [ct.c_void_p, ct.c_char_p]),
            "Snap": (ct.c_bool, [ct.c_void_p, ct.c_bool, ct.POINTER(ct.c_int),
                                  ct.POINTER(Frame), ct.c_uint]),
            "Utils_Load": (ct.c_bool, [ct.POINTER(Frame), ct.c_char_p,
                                        ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]),
            "Utils_FreeDataFrame": (None, [Frame]),
        }
        try:
            for name, (result, args) in signatures.items():
                function = getattr(api, "MPSizectorS_" + name)
                function.restype, function.argtypes = result, args
            version = float(api.MPSizectorS_GetAPIVersion())
            if abs(version - 2.776) > 0.0001:
                raise CameraError(f"SDK {version:.3f} 未验证，本接入适配 2.776")
            self.version, self.api = version, api
        except (AttributeError, CameraError):
            self._release_directories()
            raise

    def _release_directories(self):
        for directory in self._dll_directories:
            directory.close()
        self._dll_directories.clear()

    def _create(self):
        self._load()
        if not self.handle:
            self.handle = self.api.MPSizectorS_Create(0, 1)
            if not self.handle:
                raise CameraError("无法创建相机实例")
            self.api.MPSizectorS_SetAutoReconnect(self.handle, False)

    def enumerate(self):
        with self._lock:
            self._create()
            if self.device:
                raise CameraError("请先断开相机再搜索设备")
            if not self.api.MPSizectorS_UpdateDeviceList(self.handle):
                raise CameraError("搜索设备失败，请检查连接和相机驱动")
            devices = []
            count = self.api.MPSizectorS_GetDeviceCount(self.handle)
            if not 0 <= count <= 128:
                raise CameraError("SDK 返回的设备数量无效")
            for index in range(count):
                info = DeviceInfo()
                self.api.MPSizectorS_GetDeviceInfo(self.handle, ct.byref(info), index)
                devices.append(info)
            return devices

    def connect(self, serial, working_mode=2, settings_file=None, timeout_ms=10000):
        if working_mode not in WORKING_MODES:
            raise CameraError("只能选择 3D 拍摄模式")
        if not 1000 <= timeout_ms <= 120000:
            raise CameraError("超时范围应为 1～120 秒")
        with self._lock:
            self.disconnect()
            devices = self.enumerate()
            info = next((item for item in devices if item.DeviceSerialNumber == serial), None)
            if info is None:
                raise CameraError("所选相机已离线，请重新搜索")
            if info.MinimumSDKVersion > self.version + 0.0001:
                raise CameraError("相机所需 SDK 版本高于当前版本")
            if not self.api.MPSizectorS_Open(self.handle, info):
                raise CameraError("连接相机失败，请检查占用、USB 连接和驱动")
            self.device = info
            try:
                deadline = time.monotonic() + timeout_ms / 1000
                while self.api.MPSizectorS_GetDeviceState(self.handle) != 2:
                    if time.monotonic() >= deadline:
                        raise CameraError("等待相机就绪超时")
                    time.sleep(0.05)
                if settings_file:
                    file = Path(settings_file).resolve()
                    if not file.is_file():
                        raise CameraError("相机参数文件不存在")
                    if not self.api.MPSizectorS_ReadSettingsFromFile(self.handle, native_path(file)):
                        raise CameraError("导入相机参数失败")
                for name, value in (("SetHoldState", True), ("SetTriggerSource", 0),
                                    ("SetWorkingMode", working_mode), ("SetDataOutMode", 0),
                                    ("SetHoldState", False)):
                    if not getattr(self.api, "MPSizectorS_" + name)(self.handle, value):
                        raise CameraError(f"相机设置失败：{name}")
            except Exception:
                self.disconnect()
                raise
            return info

    def _read_exposure(self):
        if self.device is None or self.api.MPSizectorS_GetDeviceState(self.handle) != 2:
            raise CameraError("相机未就绪，请先连接相机")
        settings = ExposureSettings()
        if not self.api.MPSizectorS_GetExposureSettings(self.handle, ct.byref(settings)):
            raise CameraError("读取相机曝光参数失败")
        return settings

    def read_exposure(self):
        with self._lock:
            return self._read_exposure().values()

    def apply_exposure(self, values):
        values = validate_exposure(values)
        with self._lock:
            previous = self._read_exposure()
            settings = ExposureSettings.from_buffer_copy(previous)
            settings.update_values(values)
            hold = ct.c_bool()
            if not self.api.MPSizectorS_GetHoldState(self.handle, ct.byref(hold)):
                raise CameraError("读取相机暂停状态失败")
            if not self.api.MPSizectorS_SetHoldState(self.handle, True):
                raise CameraError("暂停相机失败，曝光参数未修改")
            try:
                if not self.api.MPSizectorS_SetExposureSettings(self.handle, settings):
                    raise CameraError("相机拒绝曝光参数")
                actual = self._read_exposure().values()
                if any(not math.isclose(actual[key], value, abs_tol=0.001, rel_tol=0)
                       for key, value in values.items()):
                    raise CameraError("曝光参数回读与设置不一致")
                return actual
            except Exception as error:
                if not self.api.MPSizectorS_SetExposureSettings(self.handle, previous):
                    self.disconnect()
                    raise CameraError("曝光参数恢复失败，已断开相机") from error
                raise
            finally:
                if self.device is not None and not self.api.MPSizectorS_SetHoldState(self.handle, hold.value):
                    self.disconnect()
                    raise CameraError("恢复相机暂停状态失败，已断开相机")

    def export_settings(self, path):
        path = Path(path)
        if path.suffix.lower() not in {".mpset", ".xml"} or path.exists():
            raise CameraError("请使用新的 .mpset 或 .xml 文件，不能覆盖已有参数")
        with self._lock:
            self._read_exposure()
            temporary = None
            try:
                with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
                    temporary = Path(stream.name)
                if not self.api.MPSizectorS_WriteSettingsToFile(self.handle, native_path(temporary)):
                    raise CameraError("导出相机参数失败")
                if temporary.stat().st_size == 0:
                    raise CameraError("相机参数文件为空")
                temporary.rename(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return path

    def disconnect(self):
        with self._lock:
            if self.device is not None:
                self.api.MPSizectorS_Close(self.handle)
                self.device = None

    def close(self):
        with self._lock:
            self.disconnect()
            if self.handle:
                self.api.MPSizectorS_Delete(self.handle)
                self.handle = None
            self._release_directories()

    def acquire(self, path, timeout_ms=10000, *, mpdat=None):
        if not 1000 <= timeout_ms <= 120000:
            raise CameraError("超时范围应为 1～120 秒")
        path = Path(path)
        if path.exists():
            raise CameraError(f"不能覆盖已有点云：{path}")
        with self._lock:
            self._load()
            frame, data_format = Frame(), ct.c_int(-1)
            started = time.perf_counter()
            try:
                if mpdat is not None:
                    source = Path(mpdat).resolve()
                    if not source.is_file():
                        raise CameraError("MPDAT 文件不存在")
                    if not self.api.MPSizectorS_Utils_Load(ct.byref(frame), native_path(source), None, None):
                        raise CameraError("读取 MPDAT 失败")
                else:
                    if self.device is None:
                        raise CameraError("相机未连接")
                    if self.api.MPSizectorS_GetDeviceState(self.handle) != 2:
                        raise CameraError("相机未就绪或已断开，请重新连接")
                    if not self.api.MPSizectorS_Snap(self.handle, True, ct.byref(data_format),
                                                    ct.byref(frame), timeout_ms):
                        raise CameraError("采集失败或超时，请检查曝光、连接和设备状态")
                    if data_format.value != frame.FrameInfo.DataInfo.DataFormat:
                        raise CameraError("相机返回的数据格式不一致")
                xyz, gray, image = frame_points(frame)
                info = frame.FrameInfo
                result = {"file": path.name, "points": len(xyz), "unit": "mm",
                        "width": image.shape[1], "height": image.shape[0], "image": image,
                        "serial": info.DeviceInfo.DeviceSerialNumber, "sdk_version": self.version,
                        "working_mode": ct.c_int.from_buffer_copy(bytes(info.Settings[:4])).value,
                        "captured_at": datetime.now().astimezone().isoformat(),
                        "source": str(source) if mpdat is not None else "camera",
                        "trigger_serial": int(info.DataInfo.TSN.SN)}
                # SDK 2.776 puts ExposureSettings 16 bytes after BasicSettings.
                result["exposure_settings"] = ExposureSettings.from_buffer_copy(bytes(info.Settings[16:48])).values()
                write_capture_ply(path, xyz, gray)
                result["elapsed_seconds"] = time.perf_counter() - started
                return result
            finally:
                # Snap borrows a camera-owned buffer; only Utils_Load transfers ownership.
                if mpdat is not None and frame.Data:
                    self.api.MPSizectorS_Utils_FreeDataFrame(frame)


class CaptureGroup:
    """Stage outside the watched root; publish the whole group with one directory rename."""

    def __init__(self, root, name, expected_views=DEFAULT_VIEWS, capture_settings=None):
        root = Path(root).resolve()
        name = str(name).strip()
        if (not name or len(name) > 80 or name in {".", ".."} or name.endswith((".", " "))
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
                or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                                  *[f"LPT{i}" for i in range(1, 10)]}):
            raise CameraError("组名不是有效的文件夹名称")
        if expected_views not in (5, 6, 7, 8):
            raise CameraError("每组必须有 5～8 个视角")
        if not root.is_dir() or root.parent == root:
            raise CameraError("请选择有效的非磁盘根目录作为输入目录")
        self.target = root / name
        if self.target.exists():
            raise CameraError("同名组已存在，请使用新组名")
        self.staging = root.parent / ("." + root.name + "_camera_pending") / uuid4().hex
        self.staging.mkdir(parents=True)
        self.expected_views, self.records = expected_views, []
        self.capture_settings = dict(capture_settings or {})
        self.published = False

    @property
    def next_path(self):
        if self.published or len(self.records) >= self.expected_views:
            raise CameraError("本组已采满")
        return self.staging / f"{len(self.records) + 1}.ply"

    def record(self, result):
        path = self.next_path
        if result.get("file") != path.name or not path.is_file() or path.stat().st_size == 0:
            raise CameraError("采集文件与视角顺序不一致")
        self.records.append({key: value for key, value in result.items() if key != "image"})

    def publish(self):
        if self.published or len(self.records) != self.expected_views:
            raise CameraError("只能发布采集完整的点云组")
        if self.target.exists():
            raise CameraError("同名组已存在，未覆盖")
        if sorted(path.name for path in self.staging.glob("*.ply")) != [f"{i}.ply" for i in range(1, self.expected_views + 1)]:
            raise CameraError("暂存点云缺失或包含多余视角")
        manifest = {"schema_version": 1, "expected_views": self.expected_views,
                    "unit": "mm", "capture_settings": self.capture_settings, "views": self.records}
        (self.staging / "camera_capture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.staging.rename(self.target)
        self.published = True
        return self.target
