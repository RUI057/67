#!/usr/bin/env python3
"""一鍵環境設定：建立專用的 Python 環境、裝好套件，然後開啟操作選單。

由「開始.bat」自動呼叫（Windows），一般不需要手動執行。

為什麼要專用環境，而不是直接裝進電腦上的 Python：
  - mediapipe 在 Windows 上，安裝路徑有中文（例如使用者名稱是中文）就讀不到模型檔
  - torch 的檔案路徑很深，裝在太深的資料夾會超過 Windows 260 字元的路徑上限
  - 專案若放在 OneDrive 同步的桌面，環境跟著放進去會被同步、被鎖檔而安裝失敗
  → Windows 上環境放在 C:\\Users\\Public\\cyut_venv（短、純英文、不會被同步）

流程：檢查 Python 版本 → 建立環境 → 安裝套件（只有第一次、或套件清單有變才裝）
     → 自我檢查（真的建立一次 MediaPipe 手部／臉部偵測器）→ 開啟選單 start.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

REQ    = ROOT / "requirements-core.txt"
IS_WIN = os.name == "nt"
KIT    = not (ROOT / "train_model.py").exists()     # 純收集版（make_collect_kit.py 產生）沒有訓練程式
TRAIN_MODULES = {"torch": "torch", "scikit-learn": "sklearn"}   # pip 名稱 → import 名稱；純收集版不裝
PY_MIN, PY_MAX = (3, 9), (3, 12)
PY_URL       = "https://www.python.org/downloads/release/python-3119/"
VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
STAMP_NAME   = "cyut_setup_ok.txt"
BAR = "═" * 58


class SetupError(Exception):
    """設定失敗：訊息 + 給使用者的解法。"""
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.hint = hint


def say(msg=""):
    print(msg, flush=True)


# ── Python 版本 ──────────────────────────────────────────
def check_python(v=None):
    v = tuple(v or sys.version_info[:2])
    if PY_MIN <= v <= PY_MAX:
        return
    why = ("3.13 以上太新：mediapipe 與 numpy 1.x 都還沒有對應版本。\n    "
           if v > PY_MAX else "")
    raise SetupError(f"目前用的 Python 是 {v[0]}.{v[1]}，這個專案需要 3.9～3.12。",
                     f"{why}請安裝 Python 3.11：{PY_URL}")


# ── 環境放哪裡 ───────────────────────────────────────────
def venv_python(vdir, windows=IS_WIN):
    return vdir / ("Scripts/python.exe" if windows else "bin/python")


def _writable(d):
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".cyut_write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True
    except OSError:
        return False


def venv_candidates(env=None, windows=IS_WIN):
    """依優先順序列出可放環境的位置。"""
    env = os.environ if env is None else env
    if env.get("CYUT_VENV"):                      # 進階／測試用：自訂位置
        return [Path(env["CYUT_VENV"])]
    if not windows:
        return [ROOT / ".venv"]
    cands = []
    if env.get("PUBLIC"):                          # C:\Users\Public：人人可寫、純英文
        cands.append(Path(env["PUBLIC"]) / "cyut_venv")
    cands.append(Path((env.get("SystemDrive") or "C:") + os.sep) / "cyut_venv")
    if env.get("LOCALAPPDATA"):
        cands.append(Path(env["LOCALAPPDATA"]) / "cyut_venv")
    cands.append(ROOT / ".venv")
    return cands


def pick_venv_dir(cands, windows=IS_WIN):
    """已經建過的優先（不重建）；否則挑第一個「純英文且可寫入」的位置。"""
    for d in cands:
        if venv_python(d, windows).exists():
            return d
    for d in cands:
        if str(d).isascii() and _writable(d):
            return d
    for d in cands:                                # 只剩中文路徑：先用，自我檢查失敗時會說明
        if _writable(d):
            return d
    raise SetupError("找不到可以寫入的位置來建立 Python 環境。",
                     r"確認 C:\Users\Public 可以寫入，或改用系統管理員身分執行。")


def in_venv(vdir):
    try:
        return Path(sys.prefix).resolve() == Path(vdir).resolve()
    except OSError:
        return False


def venv_works(vpy):
    if not vpy.exists():
        return False
    return subprocess.run([str(vpy), "-c", "import sys"], capture_output=True).returncode == 0


def create_venv(vdir):
    # 安全起見：資料夾已有東西、又不是 Python 環境，就不碰（避免 --clear 誤刪）
    if vdir.exists() and any(vdir.iterdir()) and not (vdir / "pyvenv.cfg").exists():
        raise SetupError(f"{vdir} 已存在而且不是 Python 環境，為了安全不會覆蓋。",
                         "請手動清空或刪除這個資料夾後再試。")
    say(f"\n建立專用 Python 環境：{vdir}")
    args = [sys.executable, "-m", "venv", str(vdir)]
    if (vdir / "pyvenv.cfg").exists():             # 壞掉的舊環境 → 重建
        args.insert(3, "--clear")
    if subprocess.call(args) != 0 or not venv_python(vdir).exists():
        raise SetupError("建立 Python 環境失敗。")


# ── 安裝套件 ─────────────────────────────────────────────
def req_fingerprint():
    return hashlib.sha256(REQ.read_bytes()).hexdigest()[:16]


def req_names():
    """requirements-core.txt 裡的套件名稱（小寫）。"""
    names = set()
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~;\[\s]", line, 1)[0].lower())
    return names


def stamp_path():
    return Path(sys.prefix) / STAMP_NAME


def already_installed():
    s = stamp_path()
    return s.exists() and s.read_text(encoding="ascii").strip() == req_fingerprint()


def install_packages():
    if not REQ.exists():
        raise SetupError("找不到 requirements-core.txt，專案檔案不完整。", "請重新下載專案。")
    say(f"\n{BAR}\n  第一次設定：下載並安裝套件\n{BAR}")
    if "torch" in req_names():
        say("需要下載約 420 MB、佔用約 1.5 GB 空間，視網路速度約 5～20 分鐘。")
    else:
        say("需要下載約 280 MB、佔用約 1 GB 空間，視網路速度約 3～10 分鐘。")
    say("畫面會一直跑文字是正常的，請不要關掉視窗。\n")
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    subprocess.call(pip + ["install", "--upgrade", "pip", "-q"])   # 舊 pip 較慢且易解析失敗；失敗不影響
    if subprocess.call(pip + ["install", "-r", str(REQ)]) != 0:
        raise SetupError(
            "套件安裝失敗。")


SMOKE = r"""
import importlib, sys
sys.path.insert(0, ".")
import numpy, cv2, PIL
import mediapipe as mp
if not hasattr(mp, "solutions"):
    raise SystemExit("MEDIAPIPE_NO_SOLUTIONS " + getattr(mp, "__version__", "?"))
extra = [importlib.import_module(m) for m in sys.argv[1:]]   # 訓練用套件，純收集版沒有
from src.features import MP_HANDS_KWARGS, MP_FACE_KWARGS
img = numpy.zeros((240, 320, 3), numpy.uint8)
with mp.solutions.hands.Hands(**MP_HANDS_KWARGS) as h:
    h.process(img)
with mp.solutions.face_mesh.FaceMesh(**MP_FACE_KWARGS) as f:
    f.process(img)
print(" / ".join(["numpy " + numpy.__version__, "opencv " + cv2.__version__,
                  "mediapipe " + mp.__version__] + [m.__name__ + " " + m.__version__ for m in extra]))
"""


def diagnose(err, prefix):
    """把自我檢查的錯誤翻成使用者看得懂的原因與解法。"""
    if any(k in err for k in ("DLL load failed", "WinError 126", "c10.dll")):
        return SetupError(
            "Windows 缺少執行元件（Microsoft Visual C++ 可轉散發套件）。",
            f"下載安裝後，重新雙擊「開始.bat」：{VCREDIST_URL}\n"
            "    或在命令提示字元執行：winget install Microsoft.VCRedist.2015+.x64")
    if "MEDIAPIPE_NO_SOLUTIONS" in err:
        return SetupError("裝到的 mediapipe 版本太新，缺少本專案需要的功能。")
    if not str(prefix).isascii():
        return SetupError("Python 環境的路徑含有中文，mediapipe 讀不到模型檔。",
                          f"刪除 {prefix} 後，把專案搬到純英文路徑（例如 C:\\CYUT_Project）再雙擊「開始.bat」。")
    return SetupError("自我檢查失敗。")


def smoke_test():
    say("\n自我檢查：實際建立 MediaPipe 手部／臉部偵測器...")
    extra = [imp for pip, imp in TRAIN_MODULES.items() if pip in req_names()]
    r = subprocess.run([sys.executable, "-c", SMOKE, *extra], capture_output=True,
                       text=True, errors="replace", cwd=str(ROOT))
    if r.returncode == 0:
        say("  ✅ " + (r.stdout.strip().splitlines() or ["OK"])[-1])
        return
    err = (r.stderr or "") + (r.stdout or "")
    say("\n".join(err.strip().splitlines()[-8:]))
    raise diagnose(err, sys.prefix)


# ── 資料夾檢查 ───────────────────────────────────────────
def check_dataset(data_dir=None, labels=None):
    """用「Download ZIP」下載、再用 Windows 內建解壓縮，中文資料夾名可能變亂碼，
    訓練就會找不到資料。回傳要顯示的警告（沒問題回傳空字串）。"""
    data_dir = Path(data_dir or ROOT / "dynamic_dataset")
    if not data_dir.is_dir():
        if KIT:
            return ""        # 純收集版本來就沒有，第一次收集時會自動建立
        return ("⚠️  找不到訓練資料夾 dynamic_dataset。\n"
                "    要訓練的話，請確認是用 git clone 下載完整專案（收集資料時會自動建立）。")
    if labels is None:
        try:
            sys.path.insert(0, str(ROOT))
            from src.vocab import get_all_labels
            labels = get_all_labels()
        except Exception:
            return ""
    labels = set(labels)
    folders = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    unknown = [f for f in folders if f not in labels]
    if folders and len(unknown) > len(folders) / 2:
        return ("⚠️  資料夾名稱和詞彙表對不上（例如：" + "、".join(unknown[:3]) + "）\n"
                "    多半是用「Download ZIP」下載、再用 Windows 內建解壓縮，中文檔名變成亂碼。\n"
                "    請改用 git clone 或 GitHub Desktop 下載專案，否則訓練會找不到資料。")
    return ""


# ── 主流程 ───────────────────────────────────────────────
def main():
    try:
        check_python()
        vdir = pick_venv_dir(venv_candidates())

        if not in_venv(vdir):
            if not venv_works(venv_python(vdir)):
                create_venv(vdir)
            # 交給專用環境裡的 Python 接手，之後裝的套件都會在那裡
            return subprocess.call([str(venv_python(vdir)), str(Path(__file__).resolve()),
                                    *sys.argv[1:]])

        # ── 以下在專用環境內執行 ──
        if not already_installed():
            install_packages()
            smoke_test()
            stamp_path().write_text(req_fingerprint(), encoding="ascii")
            say(f"\n✅ 環境設定完成（位置：{vdir}）。之後開啟會直接進入選單。")

        warn = check_dataset()
        if warn:
            say("\n" + "!" * 58 + "\n" + warn + "\n" + "!" * 58)
    except SetupError as e:
        say(f"\n❌ {e}")
        if e.hint:
            say(f"\n👉 {e.hint}")
        return 1
    except KeyboardInterrupt:
        say("\n已中斷。下次雙擊「開始.bat」會從中斷的地方繼續。")
        return 1

    return subprocess.call([sys.executable, str(ROOT / "start.py")])


if __name__ == "__main__":
    sys.exit(main())
