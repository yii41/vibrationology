# -*- coding: utf-8 -*-
"""
============================================================
2次元音場の有限差分法（FDM）シミュレーション 完全版
形状指定つき版：長方形 / 十字形 を切り替え可能
============================================================

このプログラムでできること
- 2次元格子上で音圧 P と粒子速度 U, V の時間発展を計算する
- 音源位置の音圧時刻歴を表示・保存する
- 音圧分布のスナップショットを表示・保存する
- 音場の広がりを GIF アニメーションとして保存する
- Fortran版にあったインピーダンス境界条件 FIMPE を組み込む
- 計算領域の形を「長方形」または「十字形」に設定できる
- GIF の色コントラストを入力欄で調整できる
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# 1. 入力欄（学生が主に変更する場所）
# ============================================================

# -----------------------------
# 1-1. 物理定数
# -----------------------------
RO = 1.205
C0 = 340.0

# -----------------------------
# 1-2. 計算格子
# -----------------------------
MX = 81
MY = 81
DX = 0.05
DY = 0.05

# -----------------------------
# 1-3. 時間条件
# -----------------------------
DT = 1.0e-6
NSTEP = 5000

# -----------------------------
# 1-4. 音源条件
# -----------------------------
SOURCE_TYPE = "sine"       # "sine" または "pulse"
FREQ = 271.4
SOURCE_X = MX // 2
SOURCE_Y = MY // 2

# -----------------------------
# 1-5. 計算領域の形状  ← ここを主に変更する
# -----------------------------
# "rectangle" : 長方形
# "cross"     : 十字形
DOMAIN_SHAPE = "cross"

# rectangle 用
RECT_X_MIN = 1
RECT_X_MAX = MX
RECT_Y_MIN = 1
RECT_Y_MAX = MY

# cross 用
CROSS_CENTER_X = MX // 2
CROSS_CENTER_Y = MY // 2
CROSS_HALF_WIDTH_X = 6
CROSS_HALF_WIDTH_Y = 6

CROSS_VERTICAL_Y_MIN = 1
CROSS_VERTICAL_Y_MAX = MY

CROSS_HORIZONTAL_X_MIN = 1
CROSS_HORIZONTAL_X_MAX = MX

# -----------------------------
# 1-6. 境界条件
# -----------------------------
FIMPE = 1.0e9

# -----------------------------
# 1-7. 可視化・保存条件
# -----------------------------
FRAME_INTERVAL = 50
SAVE_IMAGES = True
SAVE_GIF = True
GIF_FPS = 10

# -----------------------------
# 1-8. GIFの見た目設定
# -----------------------------
CONTRAST_SCALE = 0.3
COLORMAP_NAME = "seismic"
GAMMA = 1.0

# -----------------------------
# 1-9. 出力ファイル名
# -----------------------------
FILE_DOMAIN_NAME = "domain_shape.png"
FILE_HISTORY_NAME = "pressure_history.png"
FILE_SNAPSHOT_NAME = "pressure_snapshot.png"
FILE_GIF_NAME = "fdm2d_sound.gif"

# ============================================================
# 2. 保存先フォルダ
# ============================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

FILE_DOMAIN = os.path.join(BASE_DIR, FILE_DOMAIN_NAME)
FILE_HISTORY = os.path.join(BASE_DIR, FILE_HISTORY_NAME)
FILE_SNAPSHOT = os.path.join(BASE_DIR, FILE_SNAPSHOT_NAME)
FILE_GIF = os.path.join(BASE_DIR, FILE_GIF_NAME)

# ============================================================
# 3. 形状を作る関数
# ============================================================
def make_domain_flag(mx, my):
    lfglo = np.zeros((mx + 2, my + 2), dtype=int)

    if DOMAIN_SHAPE == "rectangle":
        x_min = max(1, RECT_X_MIN)
        x_max = min(mx, RECT_X_MAX)
        y_min = max(1, RECT_Y_MIN)
        y_max = min(my, RECT_Y_MAX)
        lfglo[x_min:x_max+1, y_min:y_max+1] = 1

    elif DOMAIN_SHAPE == "cross":
        # 縦棒
        x1 = max(1, CROSS_CENTER_X - CROSS_HALF_WIDTH_X)
        x2 = min(mx, CROSS_CENTER_X + CROSS_HALF_WIDTH_X)
        y1 = max(1, CROSS_VERTICAL_Y_MIN)
        y2 = min(my, CROSS_VERTICAL_Y_MAX)
        lfglo[x1:x2+1, y1:y2+1] = 1

        # 横棒
        x3 = max(1, CROSS_HORIZONTAL_X_MIN)
        x4 = min(mx, CROSS_HORIZONTAL_X_MAX)
        y3 = max(1, CROSS_CENTER_Y - CROSS_HALF_WIDTH_Y)
        y4 = min(my, CROSS_CENTER_Y + CROSS_HALF_WIDTH_Y)
        lfglo[x3:x4+1, y3:y4+1] = 1

    else:
        raise ValueError('DOMAIN_SHAPE は "rectangle" または "cross" にIACCFLてください。')

    return lfglo

# ============================================================
# 4. 音源位置チェック
# ============================================================
def check_source_position(lfglo, sx, sy):
    if lfglo[sx, sy] != 1:
        raise ValueError("音源位置が計算領域の外にあります。SOURCE_X, SOURCE_Y を見直してください。")

# ============================================================
# 5. 表示用の振幅変換
# ============================================================
def transform_for_display(frame, vmax_raw, gamma):
    if vmax_raw <= 0:
        return frame.copy()
    x = frame / vmax_raw
    return np.sign(x) * (np.abs(x) ** gamma) * vmax_raw

# ============================================================
# 6. 音源を加える関数
# ============================================================
def add_source(p_array, step):
    if SOURCE_TYPE == "pulse":
        itopms = 1.0
        f_ntime = 0.3
        ktmp = int(1.0 / DT / 1000.0 * itopms)
        ktmp2 = int(1.0 / DT / 1000.0 * f_ntime)
        r2 = (step - ktmp) ** 2
        ftmp = np.exp(-r2 / (float(ktmp2) ** 2))
        p_array[SOURCE_X, SOURCE_Y] += 5.0 * ftmp
    elif SOURCE_TYPE == "sine":
        p_array[SOURCE_X, SOURCE_Y] += np.sin(2.0 * np.pi * FREQ * step * DT)
    else:
        raise ValueError('SOURCE_TYPE は "sine" または "pulse" にしてください。')

# ============================================================
# 7. 補助定数の計算
# ============================================================
RK = RO * C0 * C0
COFU = DT / (DX * RO)
COFV = DT / (DY * RO)
COPX = RK * DT / DX
COPY = RK * DT / DY

# ============================================================
# 8. 配列の準備
# ============================================================
P = np.zeros((MX + 2, MY + 2), dtype=float)
U = np.zeros((MX + 1, MY + 2), dtype=float)
V = np.zeros((MX + 2, MY + 1), dtype=float)

# ============================================================
# 9. 計算領域フラグの作成
# ============================================================
lfglo = make_domain_flag(MX, MY)
check_source_position(lfglo, SOURCE_X, SOURCE_Y)

mask_center = (lfglo[1:MX+1, 1:MY+1] == 1)
mask_u = (lfglo[0:MX, 1:MY+1] == 1)
mask_v = (lfglo[1:MX+1, 0:MY] == 1)

# ============================================================
# 10. 事前表示
# ============================================================
cfl = C0 * DT * np.sqrt((1.0 / DX**2) + (1.0 / DY**2))

print("==============================================")
print("2次元音場FDMシミュレーション")
print("==============================================")
print(f"計算領域形状      : {DOMAIN_SHAPE}")
print(f"格子数            : MX = {MX}, MY = {MY}")
print(f"格子間隔          : DX = {DX} m, DY = {DY} m")
print(f"時間刻み          : DT = {DT} s")
print(f"総ステップ数      : NSTEP = {NSTEP}")
print(f"音源種類          : {SOURCE_TYPE}")
print(f"音源位置          : ({SOURCE_X}, {SOURCE_Y})")
if SOURCE_TYPE == "sine":
    print(f"音源周波数        : {FREQ} Hz")
print(f"FIMPE             : {FIMPE}")
print(f"CFLの目安値       : {cfl:.4f}")
print(f"保存先フォルダ    : {BASE_DIR}")
print("==============================================")

# ============================================================
# 11. 計算領域の形状を画像で保存
# ============================================================
plt.figure(figsize=(6, 6))
plt.imshow(lfglo[1:MX+1, 1:MY+1].T, origin="lower", aspect="equal", cmap="gray_r")
plt.scatter(SOURCE_X - 1, SOURCE_Y - 1, marker="o", s=60, label="Source")
plt.xlabel("x cell")
plt.ylabel("y cell")
plt.title(f"Domain shape: {DOMAIN_SHAPE}")
plt.legend()
plt.tight_layout()
if SAVE_IMAGES:
    plt.savefig(FILE_DOMAIN, dpi=150)
plt.show()

# ============================================================
# 12. 記録用リスト
# ============================================================
p_hist = []
u_hist = []
v_hist = []
frames = []

# ============================================================
# 13. メイン計算ループ
# ============================================================
for n in range(1, NSTEP + 1):
    add_source(P, n)

    dpx_u = P[1:MX+1, 1:MY+1] - P[0:MX, 1:MY+1]
    U[0:MX, 1:MY+1][mask_u] -= COFU * dpx_u[mask_u]

    dpy_v = P[1:MX+1, 1:MY+1] - P[1:MX+1, 0:MY]
    V[1:MX+1, 0:MY][mask_v] -= COFV * dpy_v[mask_v]

    for i in range(1, MX + 2):
        for j in range(1, MY + 2):
            if lfglo[i, j] != 0 and lfglo[i - 1, j] == 0:
                U[i - 1, j] = -P[i, j] / FIMPE
            elif lfglo[i, j] == 0 and lfglo[i - 1, j] != 0:
                U[i - 1, j] = P[i - 1, j] / FIMPE

            if lfglo[i, j] != 0 and lfglo[i, j - 1] == 0:
                V[i, j - 1] = -P[i, j] / FIMPE
            elif lfglo[i, j] == 0 and lfglo[i, j - 1] != 0:
                V[i, j - 1] = P[i, j - 1] / FIMPE

    dpx = U[1:MX+1, 1:MY+1] - U[0:MX, 1:MY+1]
    dpy = V[1:MX+1, 1:MY+1] - V[1:MX+1, 0:MY]
    P[1:MX+1, 1:MY+1][mask_center] -= COPX * dpx[mask_center] + COPY * dpy[mask_center]

    P[1:MX+1, 1:MY+1][~mask_center] = 0.0

    p_hist.append(P[SOURCE_X, SOURCE_Y])
    u_hist.append(U[SOURCE_X, SOURCE_Y])
    v_hist.append(V[SOURCE_X, SOURCE_Y])

    if n % FRAME_INTERVAL == 0:
        frames.append(P[1:MX+1, 1:MY+1].copy())

# ============================================================
# 14. 結果表示・保存
# ============================================================
t = np.arange(1, NSTEP + 1) * DT

plt.figure(figsize=(10, 4))
plt.plot(t, p_hist)
plt.xlabel("Time [s]")
plt.ylabel("Pressure at source")
plt.title("Pressure history at the source point")
plt.grid(True)
plt.tight_layout()
if SAVE_IMAGES:
    plt.savefig(FILE_HISTORY, dpi=150)
plt.show()

if len(frames) > 0:
    snapshot = frames[len(frames) // 2]
else:
    snapshot = P[1:MX+1, 1:MY+1].copy()

vmax_snapshot_raw = np.max(np.abs(snapshot))
snapshot_for_show = transform_for_display(snapshot, vmax_snapshot_raw, GAMMA)
vlim_snapshot = max(vmax_snapshot_raw * CONTRAST_SCALE, 1e-12)

plt.figure(figsize=(6, 6))
plt.imshow(
    snapshot_for_show.T,
    origin="lower",
    aspect="equal",
    cmap=COLORMAP_NAME,
    vmin=-vlim_snapshot,
    vmax=vlim_snapshot
)
plt.colorbar(label="Pressure")
plt.xlabel("x cell")
plt.ylabel("y cell")
plt.title(f"Pressure field snapshot ({DOMAIN_SHAPE})")
plt.tight_layout()
if SAVE_IMAGES:
    plt.savefig(FILE_SNAPSHOT, dpi=150)
plt.show()

if SAVE_GIF:
    if len(frames) == 0:
        print("WARNING: フレーム数が0です。FRAME_INTERVAL を小さくしてください。")
    else:
        print("GIFを生成中...")

        vmax_raw = max(np.max(np.abs(f)) for f in frames)
        if vmax_raw == 0:
            vmax_raw = 1.0

        display_frames = [transform_for_display(f, vmax_raw, GAMMA) for f in frames]
        vlim = max(vmax_raw * CONTRAST_SCALE, 1e-12)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            display_frames[0].T,
            origin="lower",
            aspect="equal",
            cmap=COLORMAP_NAME,
            vmin=-vlim,
            vmax=vlim
        )
        ax.set_xlabel("x cell")
        ax.set_ylabel("y cell")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pressure")

        def update(frame_index):
            im.set_data(display_frames[frame_index].T)
            ax.set_title(f"2D acoustic pressure field ({DOMAIN_SHAPE}) {frame_index + 1}/{len(display_frames)}")
            return [im]

        ani = FuncAnimation(fig, update, frames=len(display_frames), interval=100, blit=False)
        ani.save(FILE_GIF, writer=PillowWriter(fps=GIF_FPS))
        plt.close(fig)
        print(f"GIF保存完了: {FILE_GIF}")

print("計算が終了しました。")
print(f"最終時刻の音源位置音圧 = {p_hist[-1]:.6f}")
if SAVE_IMAGES:
    print(f"保存画像: {FILE_DOMAIN}")
    print(f"保存画像: {FILE_HISTORY}")
    print(f"保存画像: {FILE_SNAPSHOT}")
if SAVE_GIF and len(frames) > 0:
    print(f"保存GIF : {FILE_GIF}")