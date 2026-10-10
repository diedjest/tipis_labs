import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def get_spectrum(signal: np.ndarray, sampling_frequency: int, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(num_samples, 1 / sampling_frequency)[:num_samples // 2]
    amplitudes = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    return frequencies, amplitudes

def main():
    sampling_frequency = 1000
    duration = 2.0
    time = np.linspace(0, duration, int(duration * sampling_frequency), endpoint=False)
    num_samples = len(time)

    carrier_frequency = 50.0
    modulating_frequency = 4.0

    modulating_signal = 0.5 * (1 + signal.square(2 * np.pi * modulating_frequency * time))

    am_signal = modulating_signal * np.cos(2 * np.pi * carrier_frequency * time)

    frequency_deviation = 15.0
    fm_phase = 2 * np.pi * carrier_frequency * time + 2 * np.pi * frequency_deviation * np.cumsum(modulating_signal) / sampling_frequency
    fm_signal = np.cos(fm_phase)

    phase_deviation = np.pi
    pm_signal = np.cos(2 * np.pi * carrier_frequency * time + phase_deviation * modulating_signal)

    am_frequencies, am_amplitudes = get_spectrum(am_signal, sampling_frequency, num_samples)
    fm_frequencies, fm_amplitudes = get_spectrum(fm_signal, sampling_frequency, num_samples)
    pm_frequencies, pm_amplitudes = get_spectrum(pm_signal, sampling_frequency, num_samples)

    modulation_figure, modulation_axes = plt.subplots(3, 2, figsize=(14, 9))
    modulation_figure.suptitle('Амплитудная, частотная и фазовая модуляция гармонического сигнала меандром', fontsize=14)

    modulation_axes[0, 0].plot(time, am_signal, color='blue')
    modulation_axes[0, 0].set_xlim(0, 1)
    modulation_axes[0, 0].set_title('Амплитудно-модулированный сигнал (АМ)')
    modulation_axes[0, 0].set_xlabel('Время, с')
    modulation_axes[0, 0].set_ylabel('Амплитуда')
    modulation_axes[0, 0].grid(True)

    modulation_axes[0, 1].stem(am_frequencies, am_amplitudes, basefmt=" ", linefmt='blue', markerfmt='b.')
    modulation_axes[0, 1].set_xlim(0, 100)
    modulation_axes[0, 1].set_title('Спектр АМ сигнала')
    modulation_axes[0, 1].set_xlabel('Частота, Гц')
    modulation_axes[0, 1].set_ylabel('Амплитуда')
    modulation_axes[0, 1].grid(True)

    modulation_axes[1, 0].plot(time, fm_signal, color='green')
    modulation_axes[1, 0].set_xlim(0, 1)
    modulation_axes[1, 0].set_title('Частотно-модулированный сигнал (ЧМ)')
    modulation_axes[1, 0].set_xlabel('Время, с')
    modulation_axes[1, 0].set_ylabel('Амплитуда')
    modulation_axes[1, 0].grid(True)

    modulation_axes[1, 1].stem(fm_frequencies, fm_amplitudes, basefmt=" ", linefmt='green', markerfmt='g.')
    modulation_axes[1, 1].set_xlim(0, 100)
    modulation_axes[1, 1].set_title('Спектр ЧМ сигнала')
    modulation_axes[1, 1].set_xlabel('Частота, Гц')
    modulation_axes[1, 1].set_ylabel('Амплитуда')
    modulation_axes[1, 1].grid(True)

    modulation_axes[2, 0].plot(time, pm_signal, color='purple')
    modulation_axes[2, 0].set_xlim(0, 1)
    modulation_axes[2, 0].set_title('Фазово-модулированный сигнал (ФМ)')
    modulation_axes[2, 0].set_xlabel('Время, с')
    modulation_axes[2, 0].set_ylabel('Амплитуда')
    modulation_axes[2, 0].grid(True)

    modulation_axes[2, 1].stem(pm_frequencies, pm_amplitudes, basefmt=" ", linefmt='purple', markerfmt='m.')
    modulation_axes[2, 1].set_xlim(0, 100)
    modulation_axes[2, 1].set_title('Спектр ФМ сигнала')
    modulation_axes[2, 1].set_xlabel('Частота, Гц')
    modulation_axes[2, 1].set_ylabel('Амплитуда')
    modulation_axes[2, 1].grid(True)

    modulation_figure.tight_layout()
    modulation_figure.subplots_adjust(top=0.92)

    am_spectrum_full = np.fft.fft(am_signal)
    frequencies_full = np.fft.fftfreq(num_samples, 1 / sampling_frequency)

    bandwidth = 6.0
    mask = np.abs(np.abs(frequencies_full) - carrier_frequency) <= bandwidth
    distorted_spectrum = np.zeros_like(am_spectrum_full)
    distorted_spectrum[mask] = am_spectrum_full[mask]

    synthesized_signal = np.fft.ifft(distorted_spectrum).real

    synthesized_frequencies, synthesized_amplitudes = get_spectrum(synthesized_signal, sampling_frequency, num_samples)

    envelope = np.abs(signal.hilbert(synthesized_signal))
    comparator_threshold = 0.5 * (np.max(envelope) + np.min(envelope))
    recovered_signal = (envelope > comparator_threshold).astype(float)

    synthesis_figure, synthesis_axes = plt.subplots(3, 2, figsize=(14, 9))
    synthesis_figure.suptitle('Синтез, искажение и фильтрация (демодуляция) АМ сигнала', fontsize=14)

    synthesis_axes[0, 0].plot(time, modulating_signal, color='red')
    synthesis_axes[0, 0].set_xlim(0, 1)
    synthesis_axes[0, 0].set_title('Исходный модулирующий сигнал')
    synthesis_axes[0, 0].set_xlabel('Время, с')
    synthesis_axes[0, 0].set_ylabel('Амплитуда')
    synthesis_axes[0, 0].grid(True)

    synthesis_axes[0, 1].plot(time, synthesized_signal, color='orange')
    synthesis_axes[0, 1].set_xlim(0, 1)
    synthesis_axes[0, 1].set_title('Синтезированный сигнал (усеченный спектр)')
    synthesis_axes[0, 1].set_xlabel('Время, с')
    synthesis_axes[0, 1].set_ylabel('Амплитуда')
    synthesis_axes[0, 1].grid(True)

    synthesis_axes[1, 0].stem(synthesized_frequencies, synthesized_amplitudes, basefmt=" ", linefmt='orange', markerfmt='o')
    synthesis_axes[1, 0].set_xlim(0, 100)
    synthesis_axes[1, 0].set_title('Усеченный спектр АМ сигнала')
    synthesis_axes[1, 0].set_xlabel('Частота, Гц')
    synthesis_axes[1, 0].set_ylabel('Амплитуда')
    synthesis_axes[1, 0].grid(True)

    synthesis_axes[1, 1].plot(time, envelope, color='black', label='Огибающая')
    synthesis_axes[1, 1].axhline(comparator_threshold, color='magenta', linestyle='--', label='Порог компаратора')
    synthesis_axes[1, 1].set_xlim(0, 1)
    synthesis_axes[1, 1].set_title('Выделенная огибающая (преобразование Гильберта)')
    synthesis_axes[1, 1].set_xlabel('Время, с')
    synthesis_axes[1, 1].set_ylabel('Амплитуда')
    synthesis_axes[1, 1].legend()
    synthesis_axes[1, 1].grid(True)

    synthesis_axes[2, 0].plot(time, recovered_signal, color='green')
    synthesis_axes[2, 0].set_xlim(0, 1)
    synthesis_axes[2, 0].set_title('Отфильтрованный сигнал (выход компаратора)')
    synthesis_axes[2, 0].set_xlabel('Время, с')
    synthesis_axes[2, 0].set_ylabel('Амплитуда')
    synthesis_axes[2, 0].grid(True)

    synthesis_axes[2, 1].plot(time, modulating_signal, color='red', linestyle='--', label='Исходный')
    synthesis_axes[2, 1].plot(time, recovered_signal, color='green', alpha=0.7, label='Восстановленный')
    synthesis_axes[2, 1].set_xlim(0, 1)
    synthesis_axes[2, 1].set_title('Сравнение исходного и восстановленного сигналов')
    synthesis_axes[2, 1].set_xlabel('Время, с')
    synthesis_axes[2, 1].set_ylabel('Амплитуда')
    synthesis_axes[2, 1].legend()
    synthesis_axes[2, 1].grid(True)

    synthesis_figure.tight_layout()
    synthesis_figure.subplots_adjust(top=0.92)

    plt.show()

if __name__ == '__main__':
    main()