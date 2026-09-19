"""攝影機與 OpenCV 視窗的跨平台處理（收集 / 測試 / 網頁共用）。

Windows 上的兩個常見問題在這裡處理：
- 預設的 MSMF 後端開鏡頭常要等很久，甚至開了卻讀不到畫面 → 優先用 DirectShow
- 注音等中文輸入法會攔截按鍵，OpenCV 視窗收不到 N、Q 等指令 → 關閉本程式視窗的輸入法
"""
import sys

import cv2

IS_WIN = sys.platform == "win32"


def open_camera(max_index=4):
    """開啟第一個能用的攝影機；都打不開回傳 None。"""
    if not IS_WIN:
        # 維持原本行為：先試 0，打不開再試 1～3
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    # Windows：MSMF 可能「開成功卻讀不到畫面」，所以要真的讀到一幀才算數
    for idx in range(max_index):
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
    return None


def camera_help():
    """打不開攝影機時的排查步驟（依作業系統）。"""
    lines = ["打不開攝影機，請檢查：",
             "  1. 沒有其他程式（視訊會議、瀏覽器、相機 App）正在使用攝影機"]
    if IS_WIN:
        lines.append("  2. 設定 → 隱私權與安全性 → 相機 → 開啟「讓桌面應用程式存取您的相機」")
    elif sys.platform == "darwin":
        lines.append("  2. 系統設定 → 隱私權與安全性 → 攝影機 → 允許「終端機」")
    lines.append("  3. 外接攝影機請確認 USB 有接好，或拔掉重插")
    return "\n".join(lines)


def disable_ime():
    """Windows：關閉本程式視窗的輸入法，讓 OpenCV 視窗收得到 N、Q 等按鍵。

    只影響本程式自己的視窗；終端機是另一個 process，輸入詞名時照常可以打中文。
    要在第一次 cv2.imshow 之前呼叫。其他系統不做事。
    """
    if not IS_WIN:
        return
    try:
        import ctypes
        from ctypes import wintypes
        imm32 = ctypes.WinDLL("imm32")
        imm32.ImmDisableIME.argtypes = [wintypes.DWORD]
        imm32.ImmDisableIME.restype = wintypes.BOOL
        imm32.ImmDisableIME(0xFFFFFFFF)       # (DWORD)-1：本 process 的所有執行緒
    except Exception:
        pass                                   # 關不掉就算了，最多是要手動切英文


def read_key(delay=1):
    """cv2.waitKey 的包裝：英文字母一律轉小寫（大寫鎖定時也能用）。沒按鍵回傳 -1。"""
    k = cv2.waitKey(delay)
    if k < 0:
        return -1
    k &= 0xFF
    if ord("A") <= k <= ord("Z"):
        k += 32
    return k
