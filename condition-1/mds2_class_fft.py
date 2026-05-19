"""
mds2_python_commented_complete.py

目的
----

このスクリプトでは次の処理を一括で行う。
1. 多質点ばね系の運動方程式を Newmark-β 法で時間積分する
2. 各質点の変位の時間履歴を保存する
3. 代表的な質点の変位波形をグラフ化する
4. 質点運動のアニメーション GIF を作成する
5. 必要に応じて VTK ファイルを出力する

授業での使い方
--------------
- まずは main() をそのまま実行して、出力結果を確認する
- その後、質量、ばね定数、初期条件、時間刻みなどを変更し、
  応答がどう変わるかを観察する
- 外力フラグ IACCFLG を 1 に変えると、調和外力を与えることもできる
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# 1. 解析条件の設定
# ============================================================
# 時間ステップ数
ITIMESTEP = 20000

# VTK を間引いて出力するための間隔
# 100 なら 100 ステップごとに 1 ファイル出力する
IMABIKI = 100

# 質点数
IMASS = 3

# Newmark-β 法の β
# 平均加速度法では β = 1/4 を用いる
FBETA = 1.0 / 4.0

# 時間刻み [s]
DT = 0.001

# 円周率
PI = np.pi

# 記録用の質点番号
# Fortran 側の設定を踏襲している
IRECEIVE1 = IMASS   # 最後の質点
IRECEIVE2 = 1       # 1番質点
IRECEIVE3 = 1       # 1番質点（元コード互換）

# 外力フラグ
# 0: 外力なし
# 1: 調和外力あり
IACCFLG = 0

# 連立方程式の未知数総数
# 加速度 IMASS個 + 速度 IMASS個 + 変位 IMASS個 = 3*IMASS
IALL = IMASS * 3

# VTK 出力の開始ステップ
IOVER = 1


# ============================================================
# 2. VTK 出力用の関数
# ============================================================
def write_vtk(filename: Path, x: np.ndarray, mode: str = "z") -> None:
    """
    質点位置を VTK 形式で保存する。

    Parameters
    ----------
    filename : Path
        出力ファイル名
    x : np.ndarray
        各質点の変位（1始まり風に使うため x[0] は未使用）
    mode : str
        "z" なら z 方向に並べて出力
        "y" なら y 方向に変位を強調して出力

    Notes
    -----
    ParaView などで開くと、各時刻の質点位置を可視化できる。
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("4-mass spring system\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS   {IMASS}   float\n")

        if mode == "z":
            # z方向に並べる
            for i in range(1, IMASS + 1):
                f.write(f"0.0 0.0 {x[i] + i}\n")
        elif mode == "y":
            # 変位を見やすくするため 10 倍して y方向に表示
            for i in range(1, IMASS + 1):
                f.write(f"0.0 {x[i] * 10.0} {i}\n")
        else:
            raise ValueError("mode must be 'z' or 'y'.")

        # 各点を 1 頂点セルとして登録
        f.write(f"CELLS   {IMASS}   {IMASS * 2}\n")
        for k in range(IMASS):
            f.write(f"1 {k}\n")

        f.write(f"CELL_TYPES   {IMASS}\n")
        for _ in range(IMASS):
            f.write("1\n")

        # 各点のスカラー値として変位を保存
        f.write(f"POINT_DATA   {IMASS}\n")
        f.write("SCALARS point_scalars float\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(1, IMASS + 1):
            f.write(f"{x[i]}\n")


# ============================================================
# 3. シミュレーション本体
# ============================================================
def run_simulation(output_dir: str = ".", save_vtk: bool = True) -> dict:
    """
    4質点ばね系の時間応答を計算する。

    Parameters
    ----------
    output_dir : str
        出力フォルダ
    save_vtk : bool
        True のとき VTK ファイルも出力する

    Returns
    -------
    dict
        time       : 時刻配列
        history    : 全質点の変位履歴 shape=(ITIMESTEP, IMASS)
        v_001      : IRECEIVE1 の変位履歴
        v_002      : IRECEIVE2 の変位履歴
        v_003      : IRECEIVE3 の変位履歴
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 3-1. パラメータ配列の準備
    # --------------------------------------------------------
    # Fortran コードに合わせて 1 始まり風に使う
    cof_m = np.zeros(IALL + 1)   # 質量
    cof_c = np.zeros(IALL + 1)   # 減衰
    cof_k = np.zeros(IALL + 1)   # ばね定数

    # 状態量
    x = np.zeros(IMASS + 1)       # 変位
    x1dot = np.zeros(IMASS + 1)   # 速度
    x2dot = np.zeros(IMASS + 1)   # 加速度

    # 調和外力の振幅と周波数
    ampacc = np.zeros(IMASS + 1)
    freqacc = np.zeros(IMASS + 1)

    # 元コードの設定
    ampacc[IMASS] = 1.0
    freqacc[IMASS] = 1.354

    # --------------------------------------------------------
    # 3-2. 系の物性値
    # --------------------------------------------------------
    # 境界側・末端側の外側ばね/減衰
    cof_k[1] = 0.0
    cof_c[1] = 0.0
    cof_k[IMASS + 1] = 0.0
    cof_c[IMASS + 1] = 0.0

    # 内部ばね
    for i in range(2, IMASS + 1):
        cof_k[i] = 100.0
        cof_c[i] = 1.0

    # 各質点の質量
    for i in range(1, IMASS + 1):
        cof_m[i] = 1.0

    # --------------------------------------------------------
    # 3-3. 初期条件
    # --------------------------------------------------------
    # 最後の質点だけ変位 1.0 からスタート
    x[IMASS] = 1.0
    x1dot[IMASS] = 0.0

    # --------------------------------------------------------
    # 3-4. 連立方程式の係数行列を作る
    # --------------------------------------------------------
    # 未知数ベクトル root の中身は
    # [加速度4個, 速度4個, 変位4個]
    xmat2 = np.zeros((IALL, IALL))

    # --- (a) 運動方程式の質量項 M*a ---
    for i in range(1, IMASS + 1):
        xmat2[i - 1, i - 1] = cof_m[i]

    # --- (b) 減衰項 C*v ---
    ivert = 1
    ihori = 1 + IMASS
    xmat2[ivert - 1, ihori - 1] = cof_c[1] + cof_c[2]
    xmat2[ivert - 1, ihori] = -cof_c[2]

    for i in range(2, IMASS):
        ivert = i
        ihori = i + IMASS
        xmat2[ivert - 1, ihori - 2] = -cof_c[i]
        xmat2[ivert - 1, ihori - 1] = cof_c[i] + cof_c[i + 1]
        xmat2[ivert - 1, ihori] = -cof_c[i + 1]

    i = IMASS
    ivert = i
    ihori = i + IMASS
    xmat2[ivert - 1, ihori - 2] = -cof_c[i]
    xmat2[ivert - 1, ihori - 1] = cof_c[i] + cof_c[i + 1]

    # --- (c) 剛性項 K*x ---
    ivert = 1
    ihori = 1 + IMASS * 2
    xmat2[ivert - 1, ihori - 1] = cof_k[1] + cof_k[2]
    xmat2[ivert - 1, ihori] = -cof_k[2]

    for i in range(2, IMASS):
        ivert = i
        ihori = i + IMASS * 2
        xmat2[ivert - 1, ihori - 2] = -cof_k[i]
        xmat2[ivert - 1, ihori - 1] = cof_k[i] + cof_k[i + 1]
        xmat2[ivert - 1, ihori] = -cof_k[i + 1]

    i = IMASS
    ivert = i
    ihori = i + IMASS * 2
    xmat2[ivert - 1, ihori - 2] = -cof_k[i]
    xmat2[ivert - 1, ihori - 1] = cof_k[i] + cof_k[i + 1]

    # --- (d) Newmark-β 法の変位更新式 ---
    # x_(n+1) = x_n + dt*v_n + dt^2/2*a_n + β*dt^2*(a_(n+1)-a_n)
    # これを未知数 a_(n+1), x_(n+1) の形に並べ替えたもの
    for i in range(1, IMASS + 1):
        ivert = i + IMASS
        ihori = i
        xmat2[ivert - 1, ihori - 1] = -FBETA * DT ** 2

        ivert = i + IMASS
        ihori = i + IMASS
        xmat2[ivert - 1, ihori - 1] = 0.0

        ivert = i + IMASS
        ihori = i + IMASS * 2
        xmat2[ivert - 1, ihori - 1] = 1.0

    # --- (e) Newmark-β 法の速度更新式 ---
    # v_(n+1) = v_n + dt/2*(a_n + a_(n+1))
    for i in range(1, IMASS + 1):
        ivert = i + IMASS * 2
        ihori = i
        xmat2[ivert - 1, ihori - 1] = -DT / 2.0

        ivert = i + IMASS * 2
        ihori = i + IMASS
        xmat2[ivert - 1, ihori - 1] = 1.0

        ivert = i + IMASS * 2
        ihori = i + IMASS * 2
        xmat2[ivert - 1, ihori - 1] = 0.0

    # --------------------------------------------------------
    # 3-5. 時間履歴を保存する配列
    # --------------------------------------------------------
    history = []
    rec1, rec2, rec3 = [], [], []

    # --------------------------------------------------------
    # 3-6. 時間積分ループ
    # --------------------------------------------------------
    for itime in range(1, ITIMESTEP + 1):
        # 右辺ベクトル
        cof2 = np.zeros(IALL)

        # --- (a) 外力項 ---
        for i in range(1, IMASS + 1):
            if IACCFLG == 1:
                phase = float(itime) * DT * 2.0 * PI * freqacc[i]
                cof2[i - 1] = ampacc[i] * np.cos(phase)

        # --- (b) 変位更新式の右辺 ---
        for i in range(1, IMASS + 1):
            ivert = i + IMASS
            cof2[ivert - 1] = (
                DT ** 2 / 2.0 * x2dot[i]
                - FBETA * DT ** 2 * x2dot[i]
                + DT * x1dot[i]
                + x[i]
            )

        # --- (c) 速度更新式の右辺 ---
        for i in range(1, IMASS + 1):
            ivert = i + IMASS * 2
            cof2[ivert - 1] = DT / 2.0 * x2dot[i] + x1dot[i]

        # --- (d) 連立一次方程式を解く ---
        # Fortran 版では自前の消去法を使っていたが、
        # Python では NumPy の線形代数ルーチンを使う
        root = np.linalg.solve(xmat2, cof2)

        # --- (e) 解を加速度・速度・変位に戻す ---
        x2dot[1:IMASS + 1] = root[0:IMASS]
        x1dot[1:IMASS + 1] = root[IMASS:2 * IMASS]
        x[1:IMASS + 1] = root[2 * IMASS:3 * IMASS]

        # --- (f) 境界条件 ---
        # 1番質点を固定端とみなし、変位・速度を強制的に 0 にする
        x[1] = 0.0
        x1dot[1] = 0.0

        # --- (g) 履歴保存 ---
        history.append(x[1:IMASS + 1].copy())
        rec1.append(x[IRECEIVE1])
        rec2.append(x[IRECEIVE2])
        rec3.append(x[IRECEIVE3])

        # --- (h) VTK 出力 ---
        if save_vtk:
            itime2 = int((itime - IOVER) / IMABIKI)
            itmp = itime2 * IMABIKI + IOVER
            if itime == itmp and itime >= IOVER and itime2 <= 999:
                write_vtk(output_dir / f"cont{itime2}.vtk", x, mode="z")
                write_vtk(output_dir / f"cont_ver{itime2}.vtk", x, mode="y")

    # NumPy 配列へ変換
    history = np.array(history)
    rec1 = np.array(rec1)
    rec2 = np.array(rec2)
    rec3 = np.array(rec3)

    # 時刻軸
    time = np.arange(1, ITIMESTEP + 1) * DT

    # テキスト保存
    np.savetxt(output_dir / "v_001_python.txt", rec1)
    np.savetxt(output_dir / "v_002_python.txt", rec2)
    np.savetxt(output_dir / "v_003_python.txt", rec3)

    return {
        "time": time,
        "history": history,
        "v_001": rec1,
        "v_002": rec2,
        "v_003": rec3,
    }


# ============================================================
# 4. グラフ作成関数
# ============================================================
def save_time_history_plot(time: np.ndarray, history: np.ndarray, output_path: Path) -> None:
    """
    全質点の変位時刻歴を 1 枚の図に保存する。
    """
    plt.figure(figsize=(9, 5))
    for i in range(history.shape[1]):
        plt.plot(time, history[:, i], label=f"Mass {i + 1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement")
    plt.title("Time histories of all masses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ============================================================
# 5. アニメーション作成関数
# ============================================================
def save_animation_gif(
    history: np.ndarray,
    dt: float,
    output_path: Path,
    n_frames: int = 12
) -> None:
    """
    質点の変位を GIF アニメーションとして保存する。

    Parameters
    ----------
    history : np.ndarray
        shape = (時間ステップ数, 質点数)
    dt : float
        時間刻み
    output_path : Path
        出力 GIF ファイル
    n_frames : int
        GIF に使うフレーム数
    """
    # 長い解析結果をそのまま全フレームGIFにすると重くなるので、
    # 等間隔に間引いてアニメーション化する
    idx = np.linspace(0, len(history) - 1, n_frames).astype(int)
    sampled = history[idx]
    times = idx * dt

    frames = []
    y = np.arange(1, sampled.shape[1] + 1)

    # 見やすいように横軸範囲を履歴最大値から決める
    max_abs = max(1.2, float(np.max(np.abs(history))) * 1.2)

    for k in range(n_frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(sampled[k], y, marker="o")
        ax.axvline(0.0)
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(0.5, sampled.shape[1] + 0.5)
        ax.set_xlabel("Displacement")
        ax.set_ylabel("Mass index")
        ax.set_title(f"4-mass spring system   t = {times[k]:.2f} s")
        fig.tight_layout()

        # Matplotlib の描画結果を画像配列に変換
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(output_path, frames, duration=0.08, loop=0)



# ============================================================
# 6. FFT解析関数
# ============================================================
def compute_fft(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    時間波形からFFTを計算し、正の周波数側の振幅スペクトルを返す。

    Parameters
    ----------
    signal : np.ndarray
        時間波形。ここでは v_001, v_002, v_003 などを想定する。
    dt : float
        時間刻み [s]。サンプリング間隔に相当する。

    Returns
    -------
    freq : np.ndarray
        周波数軸 [Hz]
    amp : np.ndarray
        振幅スペクトル
    """
    n = len(signal)

    fft_values = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=dt)

    positive = freq >= 0
    freq = freq[positive]
    fft_values = fft_values[positive]

    amp = 2.0 * np.abs(fft_values) / n

    # 0 Hz成分だけは2倍しないよう補正する
    if len(amp) > 0:
        amp[0] = amp[0] / 2.0

    return freq, amp


def save_fft_plot(
    freq: np.ndarray,
    amp: np.ndarray,
    output_path: Path,
    title: str,
    xmax: float = 50.0
) -> None:
    """
    FFT振幅スペクトルをPNG画像として保存する。

    Parameters
    ----------
    freq : np.ndarray
        周波数軸 [Hz]
    amp : np.ndarray
        振幅スペクトル
    output_path : Path
        出力画像ファイル名
    title : str
        グラフタイトル
    xmax : float
        表示する最大周波数 [Hz]
    """
    plt.figure(figsize=(9, 5))
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.xlim(0.0, xmax)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_all_fft_results(result: dict, dt: float, output_dir: Path) -> None:
    """
    v_001, v_002, v_003 のFFTを計算し、テキストとグラフとして保存する。

    出力ファイル
    ------------
    f_001.txt : v_001のFFT結果。1列目 周波数[Hz]、2列目 振幅
    f_002.txt : v_002のFFT結果。1列目 周波数[Hz]、2列目 振幅
    f_003.txt : v_003のFFT結果。1列目 周波数[Hz]、2列目 振幅

    f_001.png : v_001の振幅スペクトル
    f_002.png : v_002の振幅スペクトル
    f_003.png : v_003の振幅スペクトル
    f_all.png : 3つの振幅スペクトルを重ね描きした図
    """
    freq1, amp1 = compute_fft(result["v_001"], dt)
    freq2, amp2 = compute_fft(result["v_002"], dt)
    freq3, amp3 = compute_fft(result["v_003"], dt)

    np.savetxt(
        output_dir / "f_001.txt",
        np.column_stack([freq1, amp1]),
        header="Frequency[Hz] Amplitude"
    )
    np.savetxt(
        output_dir / "f_002.txt",
        np.column_stack([freq2, amp2]),
        header="Frequency[Hz] Amplitude"
    )
    np.savetxt(
        output_dir / "f_003.txt",
        np.column_stack([freq3, amp3]),
        header="Frequency[Hz] Amplitude"
    )

    save_fft_plot(freq1, amp1, output_dir / "f_001.png", "FFT amplitude spectrum of v_001")
    save_fft_plot(freq2, amp2, output_dir / "f_002.png", "FFT amplitude spectrum of v_002")
    save_fft_plot(freq3, amp3, output_dir / "f_003.png", "FFT amplitude spectrum of v_003")

    plt.figure(figsize=(9, 5))
    plt.plot(freq1, amp1, label="f_001 from v_001")
    plt.plot(freq2, amp2, label="f_002 from v_002")
    plt.plot(freq3, amp3, label="f_003 from v_003")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("FFT amplitude spectra of v_001, v_002, and v_003")
    plt.xlim(0.0, 50.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "f_all.png", dpi=180)
    plt.close()


# ============================================================
# 7. 実行部
# ============================================================
def main() -> None:
    """
    解析を実行し、主要な結果ファイルを保存する。
    """
    output_dir = Path(__file__).resolve().parent / "mds2_python_complete_output"
    output_dir.mkdir(exist_ok=True)

    result = run_simulation(output_dir=output_dir, save_vtk=True)

    # 波形図
    save_time_history_plot(
        time=result["time"],
        history=result["history"],
        output_path=output_dir / "time_history_all_masses.png"
    )

    # アニメーションGIF
    save_animation_gif(
        history=result["history"],
        dt=DT,
        output_path=output_dir / "mass_motion.gif",
        n_frames=120
    )

    # FFT解析
    # v_001, v_002, v_003 の時間波形から周波数特性を求め、
    # f_001.txt, f_002.txt, f_003.txt および PNG 図として保存する。
    save_all_fft_results(
        result=result,
        dt=DT,
        output_dir=output_dir
    )

    # 画面表示用の簡単な数値出力
    print("Simulation finished.")
    print("First 5 values of v_001:")
    print(result["v_001"][:5])
    print("Last 5 values of v_001:")
    print(result["v_001"][-5:])
    print("FFT files were also saved:")
    print("  f_001.txt, f_002.txt, f_003.txt")
    print("  f_001.png, f_002.png, f_003.png, f_all.png")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()