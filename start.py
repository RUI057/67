#!/usr/bin/env python3
"""一鍵操作選單：收集資料、訓練模型、測試辨識。

給第一次接觸這個專案的人，開起來照著選單按就好。
會自動檢查並安裝各功能需要的套件。

使用方式：
  - 雙擊 「開始.bat」
  - 或終端機執行： python start.py
"""
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BAR = "═" * 58
IS_TTY = sys.stdin is not None and sys.stdin.isatty()
IS_WIN = os.name == "nt"
VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _load_specs():
    """讀 requirements-core.txt → {pip 名稱(小寫): 版本規格}。版本限制只維護那一份。"""
    specs = {}
    path = ROOT / "requirements-core.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                specs[re.split(r"[<>=!~;\[\s]", line, 1)[0].lower()] = line
    return specs


_SPECS = _load_specs()

# 各功能需要的套件：(import 名稱, pip 安裝規格)
PKG_COLLECT = [("cv2", _SPECS.get("opencv-python", "")), ("mediapipe", _SPECS.get("mediapipe", "")),
               ("PIL", _SPECS.get("pillow", "")), ("numpy", _SPECS.get("numpy", ""))]
PKG_TRAIN   = [("numpy", _SPECS.get("numpy", "")), ("torch", _SPECS.get("torch", "")),
               ("sklearn", _SPECS.get("scikit-learn", ""))]

# 在獨立 process 檢查套件：避免本程式已載入的舊模組干擾判斷，裝完後也能反映新版本
_CHECK_CODE = r"""
import importlib, sys
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        if m == "mediapipe" and not hasattr(mod, "solutions"):
            raise ImportError("mediapipe %s 太新，已移除本專案需要的 solutions API"
                              % getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m + "\t" + (type(e).__name__ + ": " + str(e)).replace("\n", " "))
"""
_OK = set()     # 本次已確認能用的套件，不用每次重查


def title(msg):
    print(f"\n{BAR}\n  {msg}\n{BAR}")


def pause():
    if IS_TTY:
        try:
            input("\n按 Enter 回到選單...")
        except (EOFError, KeyboardInterrupt):
            pass


def ask_yes(msg):
    """詢問是否繼續；直接按 Enter 視為是，讀不到輸入（EOF）視為否。"""
    try:
        return (input(f"{msg} [Y/n] ").strip().lower() or "y") in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt):
        return False


def _broken(mods):
    """回傳 {不能用的模組: 錯誤訊息}（在獨立 process 檢查）。"""
    r = subprocess.run([sys.executable, "-c", _CHECK_CODE, *mods],
                       capture_output=True, text=True, errors="replace")
    bad = {}
    for line in r.stdout.splitlines():
        m, _, err = line.partition("\t")
        if m in mods:
            bad[m] = err
    if r.returncode != 0:           # 檢查程式本身掛掉
        tail = (r.stderr.strip().splitlines() or ["未知錯誤"])[-1]
        for m in mods:
            bad.setdefault(m, tail)
    return bad


def _is_dll_error(msg):
    return any(k in msg for k in ("DLL load failed", "WinError 126", "c10.dll"))


def _print_vcredist_help():
    print("\n這是 Windows 缺少「Microsoft Visual C++ 可轉散發套件」，重裝 Python 套件沒有用。")
    print(f"   下載安裝後重新開啟：{VCREDIST_URL}")
    print("   或在命令提示字元執行：winget install Microsoft.VCRedist.2015+.x64")


def ensure_packages(pkgs, what):
    """檢查套件，缺的或版本不對的問要不要裝。回傳 True 表示可以繼續。"""
    if not _SPECS:
        print("\n找不到 requirements-core.txt，專案檔案不完整，請重新下載專案。")
        return False
    todo = [(m, s) for m, s in pkgs if m not in _OK]
    if not todo:
        return True
    bad = _broken([m for m, _ in todo])
    _OK.update(m for m, _ in todo if m not in bad)
    missing = [(m, s) for m, s in todo if m in bad]
    if not missing:
        return True

    if any(_is_dll_error(bad[m]) for m, _ in missing):
        for m, _ in missing:
            print(f"\n{m} 無法載入：{bad[m][:160]}")
        _print_vcredist_help()
        return False

    print(f"\n「{what}」需要這些套件，但還沒安裝或版本不對：")
    for mod, spec in missing:
        why = bad[mod]
        note = "" if why.startswith("ModuleNotFoundError") else f"\n      （{why[:120]}）"
        print(f"  - {spec}{note}")
    if not ask_yes("\n要現在自動安裝嗎？"):
        print("\n已取消。你也可以手動安裝：")
        print(f"  {sys.executable} -m pip install " + " ".join(s for _, s in missing))
        return False

    # -c：只裝缺的套件時，其他已裝套件（例如 numpy）也不會被相依性連帶升級到超出限制
    cmd = [sys.executable, "-m", "pip", "install", "-c", str(ROOT / "requirements-core.txt")] \
          + [s for _, s in missing]
    print(f"\n執行：{' '.join(cmd)}\n", flush=True)
    if subprocess.call(cmd) != 0:
        print("\n第一次安裝失敗，改用 --user 再試一次...\n")
        if subprocess.call(cmd + ["--user"]) != 0:
            print("\n 安裝失敗。多半是系統 Python 被保護，建議改用虛擬環境：")
            print(f"     {sys.executable} -m venv .venv")
            print("     source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)")
            print("     pip install -r requirements.txt")
            return False

    still = _broken([m for m, _ in missing])
    if still:
        for m, err in still.items():
            print(f"\n 安裝完 {m} 仍無法使用：{err[:160]}")
        if any(_is_dll_error(e) for e in still.values()):
            _print_vcredist_help()
        else:
            print("error。")
        return False
    _OK.update(m for m, _ in missing)
    print("\n套件安裝完成 ")
    return True


def data_counts():
    """回傳 {詞: 筆數}（只算格式正確的）。"""
    import numpy as np
    from src.vocab import get_all_labels
    from src.features import HANDS_DIM, FEAT_DIM

    counts = Counter()
    for label in get_all_labels():
        folder = ROOT / "dynamic_dataset" / label
        if not folder.exists():
            continue
        for f in folder.glob("*.npy"):
            try:
                shape = np.load(f, mmap_mode="r").shape
            except Exception:
                continue
            if len(shape) == 2 and shape[0] == 30 and shape[1] in (HANDS_DIM, FEAT_DIM):
                counts[label] += 1
    return counts


def show_progress():
    """顯示收集進度：哪些詞收了、哪些還沒。"""
    from src.vocab import get_all_labels
    labels = get_all_labels()
    counts = data_counts()
    done   = [l for l in labels if counts.get(l, 0) > 0]
    todo   = [l for l in labels if counts.get(l, 0) == 0]
    total  = sum(counts.values())

    title("資料收集進度")
    print(f"總樣本數：{total} 筆")
    print(f"已開始收集：{len(done)} / {len(labels)} 個詞")

    if done:
        print("\n【已收集】（筆數少的排前面，優先補）")
        for l in sorted(done, key=lambda x: counts[x]):
            mark = "  ← 偏少" if counts[l] < 30 else ""
            print(f"  {l:<8}{counts[l]:>5}{mark}")
    if todo:
        print(f"\n【還沒收集】共 {len(todo)} 個")
        for i in range(0, min(len(todo), 60), 10):
            print("  " + "、".join(todo[i:i + 10]))
        if len(todo) > 60:
            print(f"  ...（其餘 {len(todo) - 60} 個）")
    return labels, counts, todo


def do_collect():
    if not ensure_packages(PKG_COLLECT, "收集資料"):
        return
    labels, counts, todo = show_progress()

    print("\n" + "─" * 58)
    print("要從哪個詞開始收集？")
    print("  直接按 Enter = 第一個還沒收滿的詞")
    print("  或輸入「詞名」，例如：開心")
    print("  或輸入編號（上面清單的順序）")
    try:
        ans = input("\n要收集：").strip()
    except (EOFError, KeyboardInterrupt):
        return

    cmd = [sys.executable, "collect_data.py"]
    if ans:
        if ans.isdigit() and 1 <= int(ans) <= len(labels):
            cmd += ["--word", labels[int(ans) - 1]]
        elif ans in labels:
            cmd += ["--word", ans]
        else:
            print(f"\n 詞彙表裡沒有「{ans}」")
            return

    title("即將開啟攝影機")
    print("操作說明：")
    print("  手放進畫面 → 倒數 3-2-1 → 比出手勢 → 自動存檔")
    print("  N=下一個詞    M=上一個詞    J=跳到指定詞")
    print("  D=刪除上一筆  S=暫停    Q=離開")
    print("\n 第一次執行時，系統會詢問攝影機權限，請按「允許」。")
    print("   若沒有出現畫面，檢查有沒有別的程式正在用攝影機。")
    if IS_WIN:
        print("   Windows 黑畫面：設定 → 隱私權與安全性 → 相機 → 開啟「讓桌面應用程式存取您的相機」")
    print("   按鍵沒反應：先用滑鼠點一下攝影機畫面，讓它在最上層。")
    if not ask_yes("\n準備好了嗎？"):
        return
    subprocess.call(cmd)


def do_train():
    if not ensure_packages(PKG_TRAIN, "訓練模型"):
        return
    counts = data_counts()
    if not counts:
        print("\n還沒有任何訓練資料，請先選 1 收集資料。")
        return
    if len(counts) < 2:
        print(f"\n只有 1 個詞有資料（{list(counts)[0]}），至少要 2 個詞才能訓練分類模型。")
        return

    total = sum(counts.values())
    title("訓練模型")
    print(f"將用 {total} 筆資料、{len(counts)} 個詞來訓練。")
    print("\n選擇模式：")
    print("  1. 完整訓練（含交叉驗證，會算出準確率，比較久）")
    print("  2. 快速訓練（跳過驗證，只求把模型練出來）")
    try:
        mode = input("\n請選擇 [1]：").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        return

    if mode == "2":
        print("\n開始快速訓練...\n")
        rc = subprocess.call([sys.executable, "train_model.py", "--quick"])
    else:
        print("\n開始完整訓練，請耐心等待（有 GPU 約 15 分鐘，純 CPU 可能 1 小時以上）...\n")
        rc = subprocess.call([sys.executable, "train_model.py"])

    if rc == 0:
        print("\n✅ 訓練完成，模型存在 data/model.pkl")
    else:
        print("\n❌ 訓練沒有正常完成，請把上面的錯誤訊息傳給組長。")


def do_test():
    if not ensure_packages(PKG_COLLECT + PKG_TRAIN, "測試辨識"):
        return
    if not (ROOT / "data" / "model.pkl").exists():
        print("\n❌ 找不到 data/model.pkl，請先選 2 訓練模型。")
        return
    title("即將開啟攝影機測試辨識")
    print("比出手勢，畫面上會顯示辨識結果。按 Q 離開。")
    if not ask_yes("\n準備好了嗎？"):
        return
    subprocess.call([sys.executable, "test_model.py"])


def do_check():
    if not ensure_packages([("numpy", _SPECS.get("numpy", ""))], "檢查資料品質"):
        return
    subprocess.call([sys.executable, "check_dataset.py"])


def _reveal(path):
    """在檔案總管／Finder 中選取這個檔案，方便組員找到並傳送。"""
    try:
        if IS_WIN:
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
    except OSError:
        pass


def do_pack():
    """把收集到的資料打包成 zip 傳給組長。

    用程式打包而不是手動壓縮：Python 的 zip 會標記 UTF-8 檔名，
    詞資料夾（什麼、哪裡…）在任何電腦解開都不會變亂碼；組長用 merge_data.py 合併。
    """
    import zipfile
    from datetime import datetime

    data = ROOT / "dynamic_dataset"
    files = sorted(data.glob("*/*.npy")) if data.is_dir() else []
    if not files:
        print("\n❌ 還沒有收集到任何資料，請先選 1 收集資料。")
        return
    counts = Counter(f.parent.name for f in files)
    title("打包資料")
    print(f"共 {len(files)} 筆、{len(counts)} 個詞：")
    for w, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {w:<8}{n:>5}")
    try:
        name = input("\n你的名字（會放進檔名，方便組長辨認；可直接按 Enter 略過）：").strip()
    except (EOFError, KeyboardInterrupt):
        return
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name)[:20]
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = ROOT / f"手語資料_{name + '_' if name else ''}{stamp}.zip"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
        info = ROOT / "KIT_VERSION.txt"
        if info.exists():
            z.write(info, info.name)
    print(f"\n✅ 已打包：{out.name}（{out.stat().st_size / 1e6:.1f} MB）")
    print("   把這個檔案傳給組長就好（LINE、雲端硬碟都可以），不用再自己壓縮。")
    print("   原本的資料還在，之後可以繼續收；再打包一次會包含全部資料，組長合併時會自動略過重複的。")
    _reveal(out)


# 純收集版（make_collect_kit.py 產生）沒有訓練程式 → 只顯示收集相關的功能
KIT = not (ROOT / "train_model.py").exists()

if KIT:
    MENU_TITLE = "手語資料收集"
    MENU = """
  1. 收集資料      用攝影機錄製手語動作
  2. 查看進度      看哪些詞收了、哪些還沒
  3. 打包資料      收完後打包成一個檔案，傳給組長
  0. 離開
"""
else:
    MENU_TITLE = "手語辨識專案　操作選單"
    MENU = """
  1. 收集資料      用攝影機錄製手語動作
  2. 訓練模型      用收集到的資料訓練
  3. 測試辨識      用攝影機測試訓練好的模型
  4. 查看進度      看哪些詞收了、哪些還沒
  5. 檢查資料品質  找出錄壞的樣本
  0. 離開
"""


def main():
    v = sys.version_info
    if v < (3, 9):
        print(f"❌ Python 版本太舊（{v.major}.{v.minor}），需要 3.9～3.12。")
        print("👉 到 https://www.python.org/downloads/release/python-3119/ 安裝 Python 3.11。")
        pause()
        return
    if v >= (3, 13):
        # 不直接擋：已裝好套件的人仍可用「查看進度」等功能，但安裝套件一定會失敗
        print(f"⚠️  Python {v.major}.{v.minor} 太新：numpy 1.x 與 mediapipe 沒有對應版本，"
              "自動安裝套件會失敗。")
        print("   建議安裝 Python 3.11：https://www.python.org/downloads/release/python-3119/")

    if KIT:
        actions = {"1": do_collect, "2": lambda: show_progress(), "3": do_pack}
    else:
        actions = {"1": do_collect, "2": do_train, "3": do_test,
                   "4": lambda: show_progress(), "5": do_check}

    while True:
        title(MENU_TITLE)
        print(f"Python {v.major}.{v.minor}.{v.micro}　工作目錄：{ROOT.name}")
        print(MENU)
        try:
            choice = input("請輸入編號：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("0", "q", "Q"):
            print("\n掰掰！")
            return
        act = actions.get(choice)
        if not act:
            print(f"\n沒有「{choice}」這個選項，請重新輸入。")
            continue
        try:
            act()
        except KeyboardInterrupt:
            print("\n\n已中斷。")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ 發生錯誤：{type(e).__name__}: {e}")
            print("   請把完整錯誤訊息傳給組長。")
        pause()


if __name__ == "__main__":
    main()
