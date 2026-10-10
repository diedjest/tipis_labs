import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def get_spectrum(signal: np.ndarray, sampling_frequency: int, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(num_samples, 1 / sampling_frequency)[:num_samples // 2]
    amplitudes = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    return frequencies, amplitudes


def main():
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    sampling_frequency = 1000
    duration = 2.0
    time = np.linspace(0, duration, int(duration * sampling_frequency), endpoint=False)
    num_samples = len(time)

    carrier_frequency = 50.0
    modulating_frequency = 4.0

    modulating_signal = 0.5 * (1 + signal.square(2 * np.pi * modulating_frequency * time))

    am_signal = modulating_signal * np.cos(2 * np.pi * carrier_frequency * time)

    am_spectrum_full = np.fft.fft(am_signal)
    frequencies_full = np.fft.fftfreq(num_samples, 1 / sampling_frequency)

    bandwidth = 6.0
    mask = np.abs(np.abs(frequencies_full) - carrier_frequency) <= bandwidth
    distorted_spectrum = np.zeros_like(am_spectrum_full)
    distorted_spectrum[mask] = am_spectrum_full[mask]

    synthesized_signal = np.fft.ifft(distorted_spectrum).real

    analytic_signal = signal.hilbert(synthesized_signal)
    in_phase_component = analytic_signal.real
    quadrature_component = analytic_signal.imag
    squared_in_phase = in_phase_component ** 2
    squared_quadrature = quadrature_component ** 2
    sum_of_squares = squared_in_phase + squared_quadrature
    envelope = np.sqrt(sum_of_squares)

    comparator_threshold = 0.5 * (np.max(envelope) + np.min(envelope))
    digital_signal = (envelope > comparator_threshold).astype(float)

    hilbert_figure, hilbert_axes = plt.subplots(
        4, 2, figsize=(15, 9), constrained_layout=True
    )

    hilbert_axes[0, 0].plot(time, in_phase_component, color='orange')
    hilbert_axes[0, 0].set_xlim(0, 1.0)
    hilbert_axes[0, 0].set_title('1. Вещественная часть')
    hilbert_axes[0, 0].set_xlabel('Время, с')
    hilbert_axes[0, 0].set_ylabel('Амплитуда')
    hilbert_axes[0, 0].grid(True)

    hilbert_axes[0, 1].plot(time, quadrature_component, color='teal')
    hilbert_axes[0, 1].set_xlim(0, 1.0)
    hilbert_axes[0, 1].set_title('2. Мнимая часть')
    hilbert_axes[0, 1].set_xlabel('Время, с')
    hilbert_axes[0, 1].set_ylabel('Амплитуда')
    hilbert_axes[0, 1].grid(True)

    hilbert_axes[1, 0].plot(time, in_phase_component, color='orange', label='Вещественная')
    hilbert_axes[1, 0].plot(time, quadrature_component, color='teal', label='Мнимая')
    hilbert_axes[1, 0].set_xlim(0, 1.0)
    hilbert_axes[1, 0].set_title('3. Обе компоненты сигнала')
    hilbert_axes[1, 0].set_xlabel('Время, с')
    hilbert_axes[1, 0].set_ylabel('Амплитуда')
    hilbert_axes[1, 0].legend(loc='upper right')
    hilbert_axes[1, 0].grid(True)

    hilbert_axes[1, 1].plot(time, squared_in_phase, color='orange', label='Вещественная$^2$')
    hilbert_axes[1, 1].plot(time, squared_quadrature, color='teal', label='Мнимая$^2$')
    hilbert_axes[1, 1].set_xlim(0, 1.0)
    hilbert_axes[1, 1].set_title('4. Квадраты компонент')
    hilbert_axes[1, 1].set_xlabel('Время, с')
    hilbert_axes[1, 1].set_ylabel('Амплитуда$^2$')
    hilbert_axes[1, 1].legend(loc='upper right')
    hilbert_axes[1, 1].grid(True)

    hilbert_axes[2, 0].plot(time, sum_of_squares, color='brown')
    hilbert_axes[2, 0].set_xlim(0, 1.0)
    hilbert_axes[2, 0].set_title('5. Сумма квадратов')
    hilbert_axes[2, 0].set_xlabel('Время, с')
    hilbert_axes[2, 0].set_ylabel('Амплитуда$^2$')
    hilbert_axes[2, 0].grid(True)

    hilbert_axes[2, 1].plot(time, envelope, color='black', label='Огибающая')
    hilbert_axes[2, 1].plot(time, in_phase_component, color='orange', alpha=0.4, label='Вещественная')
    hilbert_axes[2, 1].set_xlim(0, 1.0)
    hilbert_axes[2, 1].set_title('6. Огибающая')
    hilbert_axes[2, 1].set_xlabel('Время, с')
    hilbert_axes[2, 1].set_ylabel('Амплитуда')
    hilbert_axes[2, 1].legend(loc='upper right')
    hilbert_axes[2, 1].grid(True)

    hilbert_axes[3, 0].plot(time, envelope, color='black', label='Огибающая')
    hilbert_axes[3, 0].axhline(comparator_threshold, color='magenta', linestyle='--', label='Порог компаратора')
    hilbert_axes[3, 0].set_xlim(0, 1.0)
    hilbert_axes[3, 0].set_title('7. Порог компаратора на огибающей')
    hilbert_axes[3, 0].set_xlabel('Время, с')
    hilbert_axes[3, 0].set_ylabel('Амплитуда')
    hilbert_axes[3, 0].legend(loc='upper right')
    hilbert_axes[3, 0].grid(True)

    hilbert_axes[3, 1].plot(time, digital_signal, color='green', label='Цифровой сигнал')
    hilbert_axes[3, 1].set_xlim(0, 1.0)
    hilbert_axes[3, 1].set_ylim(-0.2, 1.2)
    hilbert_axes[3, 1].set_title('8. Цифровой сигнал (выход компаратора)')
    hilbert_axes[3, 1].set_xlabel('Время, с')
    hilbert_axes[3, 1].set_ylabel('Уровень')
    hilbert_axes[3, 1].legend(loc='upper right')
    hilbert_axes[3, 1].grid(True)

    plt.show()


if __name__ == '__main__':
    main()