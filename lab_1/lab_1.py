import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def get_spectrum(y: np.ndarray, fs: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, 1/fs)[:n//2]
    amp = 2.0 / n * np.abs(yf[:n//2])
    return xf, amp

def main():
    fs = 1000
    t_max = 2.0
    t = np.linspace(0, t_max, int(t_max * fs), endpoint=False)
    n = len(t)
    freqs = [1, 2, 4, 8]
    max_freq_display = 45

    fig, axes = plt.subplots(len(freqs), 4, figsize=(18, 12))
    fig.suptitle('Спектры гармонических и цифровых сигналов', fontsize=16)

    for i, f in enumerate(freqs):
        y_harm = np.sin(2 * np.pi * f * t)
        xf_harm, amp_harm = get_spectrum(y_harm, fs, n)

        axes[i, 0].plot(t, y_harm, color='blue')
        axes[i, 0].set_xlim(0, 1)
        axes[i, 0].set_ylabel(f'{f} Гц\nАмплитуда')
        axes[i, 0].set_xlabel('Время, с')
        axes[i, 0].grid(True)
        if i == 0: 
            axes[i, 0].set_title('Гармонический сигнал')

        axes[i, 1].stem(xf_harm, amp_harm, basefmt=" ", linefmt='blue', markerfmt='b.')
        axes[i, 1].set_xlim(-1, max_freq_display)
        axes[i, 1].set_ylabel('Амплитуда')
        axes[i, 1].set_xlabel('Частота, Гц')
        axes[i, 1].grid(True)
        if i == 0: 
            axes[i, 1].set_title('Спектр гармонического сигнала')

        y_dig = 0.5 * (1 + signal.square(2 * np.pi * f * t))
        xf_dig, amp_dig = get_spectrum(y_dig, fs, n)
        amp_dig[0] /= 2.0 

        axes[i, 2].plot(t, y_dig, color='red')
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_ylabel('Амплитуда')
        axes[i, 2].set_xlabel('Время, с')
        axes[i, 2].grid(True)
        if i == 0: 
            axes[i, 2].set_title('Цифровой сигнал')

        axes[i, 3].stem(xf_dig, amp_dig, basefmt=" ", linefmt='red', markerfmt='r.')
        axes[i, 3].set_xlim(-1, max_freq_display)
        axes[i, 3].set_ylabel('Амплитуда')
        axes[i, 3].set_xlabel('Частота, Гц')
        axes[i, 3].grid(True)
        if i == 0: 
            axes[i, 3].set_title('Спектр цифрового сигнала')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')
    except Exception:
        try:
            manager.window.showMaximized()
        except Exception:
            pass

    plt.show()

if __name__ == '__main__':
    main()