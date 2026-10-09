import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def get_spectrum(y: np.ndarray, fs: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    amp = 2.0 / n * np.abs(yf[:n // 2])
    return xf, amp

def main():
    fs = 1000
    t_max = 2.0
    t = np.linspace(0, t_max, int(t_max * fs), endpoint=False)
    n = len(t)

    fc = 50.0
    fm = 4.0

    m = 0.5 * (1 + signal.square(2 * np.pi * fm * t))

    s_am = m * np.cos(2 * np.pi * fc * t)

    dev_f = 15.0
    phi_fm = 2 * np.pi * fc * t + 2 * np.pi * dev_f * np.cumsum(m) / fs
    s_fm = np.cos(phi_fm)

    dev_p = np.pi
    s_pm = np.cos(2 * np.pi * fc * t + dev_p * m)

    xf_am, amp_am = get_spectrum(s_am, fs, n)
    xf_fm, amp_fm = get_spectrum(s_fm, fs, n)
    xf_pm, amp_pm = get_spectrum(s_pm, fs, n)

    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 9))
    fig1.suptitle('Амплитудная, частотная и фазовая модуляция гармонического сигнала меандром', fontsize=14)

    axes1[0, 0].plot(t, s_am, color='blue')
    axes1[0, 0].set_xlim(0, 1)
    axes1[0, 0].set_title('Амплитудно-модулированный сигнал (АМ)')
    axes1[0, 0].set_xlabel('Время, с')
    axes1[0, 0].set_ylabel('Амплитуда')
    axes1[0, 0].grid(True)

    axes1[0, 1].stem(xf_am, amp_am, basefmt=" ", linefmt='blue', markerfmt='b.')
    axes1[0, 1].set_xlim(0, 100)
    axes1[0, 1].set_title('Спектр АМ сигнала')
    axes1[0, 1].set_xlabel('Частота, Гц')
    axes1[0, 1].set_ylabel('Амплитуда')
    axes1[0, 1].grid(True)

    axes1[1, 0].plot(t, s_fm, color='green')
    axes1[1, 0].set_xlim(0, 1)
    axes1[1, 0].set_title('Частотно-модулированный сигнал (ЧМ)')
    axes1[1, 0].set_xlabel('Время, с')
    axes1[1, 0].set_ylabel('Амплитуда')
    axes1[1, 0].grid(True)

    axes1[1, 1].stem(xf_fm, amp_fm, basefmt=" ", linefmt='green', markerfmt='g.')
    axes1[1, 1].set_xlim(0, 100)
    axes1[1, 1].set_title('Спектр ЧМ сигнала')
    axes1[1, 1].set_xlabel('Частота, Гц')
    axes1[1, 1].set_ylabel('Амплитуда')
    axes1[1, 1].grid(True)

    axes1[2, 0].plot(t, s_pm, color='purple')
    axes1[2, 0].set_xlim(0, 1)
    axes1[2, 0].set_title('Фазово-модулированный сигнал (ФМ)')
    axes1[2, 0].set_xlabel('Время, с')
    axes1[2, 0].set_ylabel('Амплитуда')
    axes1[2, 0].grid(True)

    axes1[2, 1].stem(xf_pm, amp_pm, basefmt=" ", linefmt='purple', markerfmt='m.')
    axes1[2, 1].set_xlim(0, 100)
    axes1[2, 1].set_title('Спектр ФМ сигнала')
    axes1[2, 1].set_xlabel('Частота, Гц')
    axes1[2, 1].set_ylabel('Амплитуда')
    axes1[2, 1].grid(True)

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.92)

    yf_am_full = np.fft.fft(s_am)
    freqs_full = np.fft.fftfreq(n, 1 / fs)

    bandwidth = 6.0
    mask = np.abs(np.abs(freqs_full) - fc) <= bandwidth
    yf_distorted = np.zeros_like(yf_am_full)
    yf_distorted[mask] = yf_am_full[mask]

    s_synth = np.fft.ifft(yf_distorted).real

    xf_dist, amp_dist = get_spectrum(s_synth, fs, n)

    envelope = np.abs(signal.hilbert(s_synth))
    comparator_threshold = 0.5 * (np.max(envelope) + np.min(envelope))
    s_recovered = (envelope > comparator_threshold).astype(float)

    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 9))
    fig2.suptitle('Синтез, искажение и фильтрация (демодуляция) АМ сигнала', fontsize=14)

    axes2[0, 0].plot(t, m, color='red')
    axes2[0, 0].set_xlim(0, 1)
    axes2[0, 0].set_title('Исходный модулирующий сигнал')
    axes2[0, 0].set_xlabel('Время, с')
    axes2[0, 0].set_ylabel('Амплитуда')
    axes2[0, 0].grid(True)

    axes2[0, 1].plot(t, s_synth, color='orange')
    axes2[0, 1].set_xlim(0, 1)
    axes2[0, 1].set_title('Синтезированный сигнал (усеченный спектр)')
    axes2[0, 1].set_xlabel('Время, с')
    axes2[0, 1].set_ylabel('Амплитуда')
    axes2[0, 1].grid(True)

    axes2[1, 0].stem(xf_dist, amp_dist, basefmt=" ", linefmt='orange', markerfmt='o')
    axes2[1, 0].set_xlim(0, 100)
    axes2[1, 0].set_title('Усеченный спектр АМ сигнала')
    axes2[1, 0].set_xlabel('Частота, Гц')
    axes2[1, 0].set_ylabel('Амплитуда')
    axes2[1, 0].grid(True)

    axes2[1, 1].plot(t, envelope, color='black', label='Огибающая')
    axes2[1, 1].axhline(comparator_threshold, color='magenta', linestyle='--', label='Порог компаратора')
    axes2[1, 1].set_xlim(0, 1)
    axes2[1, 1].set_title('Выделенная огибающая (преобразование Гильберта)')
    axes2[1, 1].set_xlabel('Время, с')
    axes2[1, 1].set_ylabel('Амплитуда')
    axes2[1, 1].legend()
    axes2[1, 1].grid(True)

    axes2[2, 0].plot(t, s_recovered, color='green')
    axes2[2, 0].set_xlim(0, 1)
    axes2[2, 0].set_title('Отфильтрованный сигнал (выход компаратора)')
    axes2[2, 0].set_xlabel('Время, с')
    axes2[2, 0].set_ylabel('Амплитуда')
    axes2[2, 0].grid(True)

    axes2[2, 1].plot(t, m, color='red', linestyle='--', label='Исходный')
    axes2[2, 1].plot(t, s_recovered, color='green', alpha=0.7, label='Восстановленный')
    axes2[2, 1].set_xlim(0, 1)
    axes2[2, 1].set_title('Сравнение исходного и восстановленного сигналов')
    axes2[2, 1].set_xlabel('Время, с')
    axes2[2, 1].set_ylabel('Амплитуда')
    axes2[2, 1].legend()
    axes2[2, 1].grid(True)

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92)

    plt.show()

if __name__ == '__main__':
    main()