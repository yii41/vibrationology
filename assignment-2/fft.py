import numpy as np
import matplotlib.pyplot as plt

# ==================================
# ファイル名
# ==================================

input_csv = "beam.csv"
output_csv = "fft_result.csv"

# ==================================
# FFT設定
# ==================================

Nfft = 131072

fmax = 1000

dynamic_range = 100

# ==================================
# CSV読込
# ==================================

data = np.loadtxt(input_csv, delimiter=",")

t = data[:,0]
x = data[:,1]

# ==================================
# 平均値除去
# ==================================

x = x - np.mean(x)

# ==================================
# サンプリング条件
# ==================================

dt = t[1] - t[0]
fs = 1.0 / dt

print("Sampling Frequency =", fs, "Hz")
print("Number of Samples  =", len(x))

# ==================================
# FFT
# ==================================

Y = np.fft.rfft(x, n=Nfft)

freq = np.fft.rfftfreq(Nfft, d=dt)

amp = 2*np.abs(Y)/len(x)

amp_db = 20*np.log10(amp + 1e-20)

# ==================================
# FFT結果CSV保存
# ==================================

fft_result = np.column_stack(
    (
        freq,
        amp,
        amp_db
    )
)

np.savetxt(
    output_csv,
    fft_result,
    delimiter=",",
    header="Frequency_Hz,Amplitude,Level_dB",
    comments=""
)

print()
print("FFT result saved to")
print(output_csv)

# ==================================
# 最大ピーク
# ==================================

idx = np.argmax(amp[1:]) + 1

print()
print("Dominant Frequency")
print("------------------")
print(f"{freq[idx]:.3f} Hz")

# ==================================
# グラフ表示
# ==================================

peak = np.max(amp_db)

plt.figure(figsize=(10,5))

plt.plot(freq, amp_db)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Level [dB]")

plt.title("FFT Spectrum")

plt.grid(True)

plt.xlim(0, fmax)

plt.ylim(
    peak-dynamic_range,
    peak+3
)

plt.tight_layout()

plt.show()