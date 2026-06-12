"""
最小海康相机取流验证 — MV-CS050-10UC

只做一件事: Python 使用 Hikrobot MVS SDK 打开相机，用 cv2.imshow 显示实时画面。

Usage:
  python tools/test_hik_camera_minimal.py

前提:
  1. 关闭 MVS 软件（相机不能被 MVS 占用）
  2. 相机通过 USB3 连接
  3. MVS SDK 安装在 D:/develop/MVS
"""
import sys
import os
import ctypes
import time
import numpy as np
from pathlib import Path
from ctypes import c_uint, c_void_p, c_bool, byref, cast, POINTER, memset, sizeof

# ── Step 1: 添加 DLL 搜索路径 ──
_MVS_ROOT = r"D:\develop\MVS"
_MVS_SDK_PYTHON = os.path.join(_MVS_ROOT, "Development", "Samples", "Python")
_MVS_MV_IMPORT = os.path.join(_MVS_SDK_PYTHON, "MvImport")

# Python 3.8+ 需要显式添加 DLL 目录
_MVS_LIB_PATHS = [
    os.path.join(_MVS_ROOT, "Development", "Libraries", "win64"),
    os.path.join(_MVS_ROOT, "Runtime", "win64"),
    r"C:\Program Files (x86)\MVS\Development\Libraries\win64",
]
for p in _MVS_LIB_PATHS:
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
            print(f"[DLL dir] Added: {p}")
        except Exception:
            pass

sys.path.insert(0, _MVS_SDK_PYTHON)
sys.path.insert(0, _MVS_MV_IMPORT)

from MvCameraControl_class import MvCamera, MV_CC_DEVICE_INFO_LIST, MVCC_INTVALUE, MVCC_FLOATVALUE, MVCC_ENUMVALUE, MV_FRAME_OUT
from CameraParams_header import MV_CC_DEVICE_INFO, MV_USB_DEVICE, MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE
from MvErrorDefine_const import MV_OK
import PixelType_header as px


def decoding_char(c_ubyte_value) -> str:
    """解码 SDK 返回的 c_ubyte 数组为中文字符串"""
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        return c_char_p_value.value.decode('gbk')
    except (UnicodeDecodeError, AttributeError):
        try:
            return c_char_p_value.value.decode('utf-8')
        except Exception:
            return str(c_char_p_value.value) if c_char_p_value.value else ""


def to_hex_str(num: int) -> str:
    """错误码转十六进制字符串"""
    cha_dic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    if num < 0:
        num = num + 2 ** 32
    hex_str = ""
    while num >= 16:
        digit = num % 16
        hex_str = cha_dic.get(digit, str(digit)) + hex_str
        num //= 16
    hex_str = cha_dic.get(num, str(num)) + hex_str
    return "0x" + hex_str


# 全局变量，用于只打印一次像素格式
_PRINTED_FORMAT = None


def convert_frame_to_bgr(st_frame) -> np.ndarray:
    """将 SDK 图像帧转换为 OpenCV BGR 格式 (uint8, HxWx3)"""
    global _PRINTED_FORMAT
    import cv2

    fi = st_frame.stFrameInfo
    w, h = fi.nWidth, fi.nHeight
    fmt = fi.enPixelType
    buf_addr = st_frame.pBufAddr
    payload = st_frame.stFrameInfo.nFrameLen

    # 只在格式变化时打印
    if _PRINTED_FORMAT != fmt:
        print(f"[Frame] {w}x{h} 像素格式={to_hex_str(fmt)}"
              f"({'BayerRG8' if fmt == px.PixelType_Gvsp_BayerRG8 else ''}"
              f"{'BayerGB8' if fmt == px.PixelType_Gvsp_BayerGB8 else ''}"
              f"{'BayerGR8' if fmt == px.PixelType_Gvsp_BayerGR8 else ''}"
              f"{'BayerBG8' if fmt == px.PixelType_Gvsp_BayerBG8 else ''}"
              f"{'Mono8' if fmt == px.PixelType_Gvsp_Mono8 else ''}"
              f"{'RGB8' if fmt == px.PixelType_Gvsp_RGB8_Packed else ''}"
              f"{'BGR8' if fmt == px.PixelType_Gvsp_BGR8_Packed else ''}"
              f") payload={payload}")
        _PRINTED_FORMAT = fmt

    # ── Mono8 ──
    if fmt == px.PixelType_Gvsp_Mono8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        gray = raw.reshape((h, w))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── RGB8 Packed ──
    if fmt == px.PixelType_Gvsp_RGB8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
        rgb = raw.reshape((h, w, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── BGR8 Packed ──
    if fmt == px.PixelType_Gvsp_BGR8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
        return raw.reshape((h, w, 3))

    # ── BayerRG8 ──
    if fmt == px.PixelType_Gvsp_BayerRG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        bayer = raw.reshape((h, w))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)

    # ── BayerGB8 ──
    if fmt == px.PixelType_Gvsp_BayerGB8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        bayer = raw.reshape((h, w))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)

    # ── BayerGR8 ──
    if fmt == px.PixelType_Gvsp_BayerGR8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        bayer = raw.reshape((h, w))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)

    # ── BayerBG8 ──
    if fmt == px.PixelType_Gvsp_BayerBG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        bayer = raw.reshape((h, w))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

    # ── BayerRG10 / BayerRG10_Packed etc. ──
    if fmt == px.PixelType_Gvsp_BayerRG10:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * payload)).contents)
        # 10-bit unpacked → 16-bit → demosaic
        arr = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))
        arr_8bit = (arr >> 2).astype(np.uint8)
        return cv2.cvtColor(arr_8bit, cv2.COLOR_BayerRG2BGR)

    # ── Fallback: try raw reshape ──
    print(f"[Warning] Unknown pixel format {to_hex_str(fmt)}, trying raw reshape...")
    try:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * payload)).contents)
        return raw.reshape((h, w, 3))
    except Exception:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)


def main():
    import cv2

    print("=" * 60)
    print("海康相机最小取流验证 — MV-CS050-10UC")
    print("=" * 60)

    # ── 提醒用户 ──
    print("\n⚠  运行前请确认:")
    print("  1. MVS 软件已关闭")
    print("  2. 相机 USB3 线已连接")
    print("  3. 相机指示灯正常")
    print()

    cam = None
    grabbing = False

    try:
        # ── Step 2: 初始化 SDK ──
        print("[1/7] 初始化 MVS SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            print(f"  FAILED: ret={to_hex_str(ret)}")
            return 1
        print("  OK")

        # ── Step 3: 枚举设备 ──
        print("[2/7] 枚举设备 (USB + GigE)...")
        device_list = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
        if ret != MV_OK:
            print(f"  FAILED: ret={to_hex_str(ret)}")
            return 1

        n_devices = device_list.nDeviceNum
        print(f"  找到 {n_devices} 个设备")

        if n_devices == 0:
            print("  ✗ 未找到任何设备！")
            print("  检查: MVS 是否关闭? USB 线是否插在 USB3 口?")
            return 1

        # ── Step 4: 列出所有设备，优先选择 USB ──
        selected_idx = -1
        target_serial = "DA9562204"

        for i in range(n_devices):
            mvcc_dev_info = cast(device_list.pDeviceInfo[i],
                                 POINTER(MV_CC_DEVICE_INFO)).contents
            layer_type = mvcc_dev_info.nTLayerType

            if layer_type == MV_USB_DEVICE:
                info = mvcc_dev_info.SpecialInfo.stUsb3VInfo
                model = decoding_char(info.chModelName)
                serial = ""
                for ch in info.chSerialNumber:
                    if ch == 0:
                        break
                    serial += chr(ch)
                user_name = decoding_char(info.chUserDefinedName)
                print(f"  [{i}] USB: {model} | SN:{serial} | Name:{user_name}")

                if serial == target_serial:
                    selected_idx = i
                    print(f"       ^^^ 目标设备！")

                if selected_idx < 0:
                    selected_idx = i  # 第一个 USB 设备兜底

            elif layer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                info = mvcc_dev_info.SpecialInfo.stGigEInfo
                model = decoding_char(info.chModelName)
                user_name = decoding_char(info.chUserDefinedName)
                print(f"  [{i}] GigE: {model} | Name:{user_name}")

        if selected_idx < 0:
            print("  ✗ 未找到 USB 相机！")
            return 1

        print(f"\n  选择设备索引: {selected_idx}")

        # ── Step 5: 创建句柄并打开设备 ──
        print("[3/7] 打开设备...")
        st_device = cast(device_list.pDeviceInfo[selected_idx],
                         POINTER(MV_CC_DEVICE_INFO)).contents

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(st_device)
        if ret != MV_OK:
            print(f"  CreateHandle FAILED: {to_hex_str(ret)}")
            return 1

        ret = cam.MV_CC_OpenDevice()
        if ret != MV_OK:
            print(f"  OpenDevice FAILED: {to_hex_str(ret)}")
            print(f"  → 检查 MVS 软件是否已关闭")
            cam.MV_CC_DestroyHandle()
            return 1
        print("  OK — 设备已打开")

        # ── Step 6: 读取当前参数（保留 MVS 软件的配置）+ 只改必要的 ──
        print("[4/7] 读取/设置采集参数...")

        # 最关键: TriggerMode = Off（连续采集模式）
        ret = cam.MV_CC_SetEnumValue("TriggerMode", 0)
        if ret != MV_OK:
            print(f"  TriggerMode 设置失败: {to_hex_str(ret)} (继续...)")
        else:
            print("  TriggerMode = Off ✓")

        # ── 读取当前 MVS 配置的参数（不要乱改！）──
        st_exposure = MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("ExposureTime", st_exposure) == MV_OK:
            print(f"  [当前] ExposureTime = {st_exposure.fCurValue:.1f} us")
        else:
            st_exposure.fCurValue = 40000.0

        st_gain = MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("Gain", st_gain) == MV_OK:
            print(f"  [当前] Gain = {st_gain.fCurValue:.2f}")
        else:
            st_gain.fCurValue = 0.0

        # 自动曝光状态
        st_auto_exp = MVCC_ENUMVALUE()
        memset(byref(st_auto_exp), 0, sizeof(MVCC_ENUMVALUE))
        if cam.MV_CC_GetEnumValue("ExposureAuto", st_auto_exp) == MV_OK:
            print(f"  [当前] ExposureAuto = {st_auto_exp.nCurValue} (0=Off, 1=Once, 2=Continuous)")
        else:
            print("  [当前] ExposureAuto = ? (读取失败)")

        # 自动增益状态
        st_auto_gain = MVCC_ENUMVALUE()
        memset(byref(st_auto_gain), 0, sizeof(MVCC_ENUMVALUE))
        if cam.MV_CC_GetEnumValue("GainAuto", st_auto_gain) == MV_OK:
            print(f"  [当前] GainAuto = {st_auto_gain.nCurValue} (0=Off, 1=Once, 2=Continuous)")
        else:
            print("  [当前] GainAuto = ? (读取失败)")

        # ── 如果自动曝光和自动增益都是 Off 但画面暗，保持 MVS 原来的曝光值即可 ──
        # 关键修复: 不用硬编码的 12000，保持相机已有的配置！
        current_exposure = st_exposure.fCurValue
        if current_exposure < 1000:
            # 曝光值异常低，设一个合理的默认值
            current_exposure = 40000.0
            cam.MV_CC_SetFloatValue("ExposureTime", current_exposure)
            print(f"  → 曝光值异常，设为 {current_exposure:.0f} us")

        # 如果曝光 < 15000，适当提高
        if current_exposure < 15000:
            new_exposure = 35000.0
            ret = cam.MV_CC_SetFloatValue("ExposureTime", new_exposure)
            if ret == MV_OK:
                print(f"  → 曝光从 {current_exposure:.0f} 提高到 {new_exposure:.0f} us")
                current_exposure = new_exposure
            else:
                print(f"  → 设置曝光失败: {to_hex_str(ret)}，保持 {current_exposure:.0f} us")

        # 帧率
        st_fps = MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("ResultingFrameRate", st_fps) == MV_OK:
            print(f"  [当前] FrameRate = {st_fps.fCurValue:.1f} fps")

        # 分辨率
        st_w = MVCC_INTVALUE()
        st_h = MVCC_INTVALUE()
        cam.MV_CC_GetIntValue("Width", st_w)
        cam.MV_CC_GetIntValue("Height", st_h)
        print(f"  [当前] 分辨率 = {st_w.nCurValue}x{st_h.nCurValue}")

        # 像素格式
        st_pixel = MVCC_ENUMVALUE()
        memset(byref(st_pixel), 0, sizeof(MVCC_ENUMVALUE))
        if cam.MV_CC_GetEnumValue("PixelFormat", st_pixel) == MV_OK:
            pf_val = st_pixel.nCurValue
            pf_name = {0x02180014: "BayerRG8", 0x02180015: "BayerGR8",
                       0x02180016: "BayerGB8", 0x02180017: "BayerBG8",
                       0x01080001: "Mono8", 0x0210001f: "RGB8Packed",
                       0x02180013: "BGR8Packed"}.get(pf_val, "?")
            print(f"  [当前] PixelFormat = {to_hex_str(pf_val)} ({pf_name})")

        # ── Step 7: 开始取流 ──
        print("[5/7] 开始取流...")
        ret = cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            print(f"  StartGrabbing FAILED: {to_hex_str(ret)}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return 1
        grabbing = True
        print("  OK — 取流已开始")

        # ── 主循环: 取帧并显示 ──
        print("[6/7] 显示实时画面 (按 q 或 ESC 退出)")
        print("=" * 60)

        frame_count = 0
        fps_times = []
        start_time = time.time()

        cv2.namedWindow("HikCamera MV-CS050-10UC", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HikCamera MV-CS050-10UC", 960, 720)

        while True:
            st_frame = MV_FRAME_OUT()
            memset(byref(st_frame), 0, sizeof(MV_FRAME_OUT))

            ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
            if ret != MV_OK or st_frame.pBufAddr is None:
                # 无新帧，继续等待
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue

            try:
                frame = convert_frame_to_bgr(st_frame)
                frame_count += 1

                # FPS 计算
                fps_times.append(time.time())
                if len(fps_times) > 30:
                    fps_times = fps_times[-30:]
                fps = 0.0
                if len(fps_times) >= 2:
                    dt = fps_times[-1] - fps_times[0]
                    fps = (len(fps_times) - 1) / dt if dt > 0 else 0

                # 叠加信息
                h_disp, w_disp = frame.shape[:2]
                cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press q or ESC to quit",
                            (10, h_disp - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

                cv2.imshow("HikCamera MV-CS050-10UC", frame)

            finally:
                cam.MV_CC_FreeImageBuffer(st_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        elapsed = time.time() - start_time
        print(f"\n[6/7] 共采集 {frame_count} 帧，耗时 {elapsed:.1f}s，"
              f"平均帧率 {frame_count / elapsed:.1f} fps")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # ── 清理 ──
        print("[7/7] 清理资源...")
        cv2.destroyAllWindows()

        if cam is not None:
            if grabbing:
                try:
                    ret = cam.MV_CC_StopGrabbing()
                    print(f"  StopGrabbing: {'OK' if ret == MV_OK else to_hex_str(ret)}")
                except Exception as e:
                    print(f"  StopGrabbing error: {e}")
                grabbing = False

            try:
                ret = cam.MV_CC_CloseDevice()
                print(f"  CloseDevice: {'OK' if ret == MV_OK else to_hex_str(ret)}")
            except Exception as e:
                print(f"  CloseDevice error: {e}")

            try:
                ret = cam.MV_CC_DestroyHandle()
                print(f"  DestroyHandle: {'OK' if ret == MV_OK else to_hex_str(ret)}")
            except Exception as e:
                print(f"  DestroyHandle error: {e}")

        try:
            MvCamera.MV_CC_Finalize()
            print("  SDK Finalize: OK")
        except Exception:
            pass

        print("\nDone.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
