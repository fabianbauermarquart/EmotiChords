from typing import Callable, List

import pygame, pygame.sndarray
import numpy as np
import scipy.signal
import scipy.fftpack
import librosa

from constants import intervals, chords, advanced_emotichords, notes

sampling_rate = 44100

pygame.init()

length = 1000


def play_for(sample_wave: np.array, duration: int):
    """
    Play the given numpy array, as a sound, for duration milliseconds.

    :param sample_wave: Sample wave as a numpy array.
    :param duration:    Duration [ms].
    :return:
    """
    sound = pygame.sndarray.make_sound(sample_wave)
    sound.play(-1)
    pygame.time.delay(duration)
    sound.stop()


def load_signal_wav(name: str, stereo: bool = False) -> np.array:
    signal, _ = librosa.load(name + '.wav', sr=sampling_rate, mono=True)

    if stereo:
        signal = np.repeat(signal.reshape(signal.shape[0], 1), 2, axis=1)

    return signal


def synthesize_wave() -> np.array:
    def find_peaks(frequencies, amplitudes, width, lookaround):
        peak_indices = scipy.signal.find_peaks_cwt(amplitudes, widths=(width,))
        amplitudes_maxima = list(map(lambda idx: np.max(amplitudes[idx - lookaround:idx + lookaround]), peak_indices))
        frequencies_maxima = frequencies[np.isin(amplitudes, amplitudes_maxima)]
        return frequencies_maxima, amplitudes_maxima

    def compose_sine_waves(frequencies, amplitudes, duration):
        return sum(map(lambda fa: sine_wave(fa[0], 2) * fa[1], zip(frequencies, amplitudes)))

    samples_original = load_signal_wav('violin')
    N = samples_original.shape[0]
    spectrum = scipy.fftpack.fft(samples_original)
    frequencies = scipy.fftpack.fftfreq(N, 1 / sampling_rate)
    frequencies = frequencies[:N // 2]
    amplitudes = np.abs(spectrum[:N // 2])

    frequencies_maxima, amplitudes_maxima = find_peaks(frequencies, amplitudes, 60, 10)

    samples_reconstructed = compose_sine_waves(frequencies_maxima, amplitudes_maxima, 2)

    return samples_reconstructed


def sine_wave(frequency: float, peak_amplitude: int, num_samples=sampling_rate) -> np.array:
    """
    Compute n samples of a sine wave with given frequency and peak amplitude.
    Defaults to one second.

    :param frequency:       Frequency [Hz].
    :param peak_amplitude:  Peak amplitude.
    :param num_samples:     Number of samples.
    :return: Sine wave.
    """
    length = sampling_rate / float(frequency)
    omega = np.pi * 2 / length
    x_values = np.arange(int(length)) * omega
    one_cycle = peak_amplitude * np.sin(x_values)

    mono_sine = np.resize(one_cycle, (num_samples,)).astype(np.int16)
    stereo_sine = np.repeat(mono_sine.reshape(num_samples, 1), 2, axis=1)

    return stereo_sine


def square_wave(frequency: float, peak_amplitude: int, duty_cycle=.5, num_samples=sampling_rate) -> np.array:
    """
    Compute N samples of a square wave with given frequency and peak amplitude.
    Defaults to one second.

    :param frequency:       Frequency [Hz].
    :param peak_amplitude:  Peak amplitude.
    :param duty_cycle:      Square wave duty cycle.
    :param num_samples:     Number of samples.
    :return: Square wave.
    """
    t = np.linspace(0, 1, int(500 * 440 / frequency), endpoint=False)
    wave = scipy.signal.square(2 * np.pi * 5 * t, duty=duty_cycle)
    wave = np.resize(wave, (num_samples,))

    wave = peak_amplitude / 2 * wave
    wave = np.repeat(wave.reshape(num_samples, 1), 2, axis=1)

    return wave.astype(np.int16)


def sawtooth_wave(frequency: float, peak_amplitude: int, num_samples=sampling_rate) -> np.array:
    """
    Compute N samples of a sawtooth wave with given frequency and peak amplitude.
    Defaults to one second.

    :param frequency:       Frequency [Hz].
    :param peak_amplitude:  Peak amplitude.
    :param num_samples:     Number of samples.
    :return: Square wave.
    """
    t = np.linspace(0, 1, int(500 * 440 / frequency), endpoint=False)
    wave = scipy.signal.sawtooth(2 * np.pi * 5 * t)
    wave = np.resize(wave, (num_samples,))

    wave = peak_amplitude / 2 * wave
    wave = np.repeat(wave.reshape(num_samples, 1), 2, axis=1)

    return wave.astype(np.int16)


def make_chord(frequency: float, ratios: List[List[int]], waveform: Callable = None) -> np.array:
    """
    Make a chord based on a list of frequency ratios using a given waveform (defaults to a sine wave).

    :param frequency:   Frequency [Hz].
    :param ratios:      Ratios between base note (tonic) to chord notes.
    :param waveform:    Optional waveform, with frequency [Hz] and peak amplitude parameters.
    :return:
    """
    sampling_rate = 4096

    if not waveform:
        waveform = sine_wave

    chord = waveform(frequency, sampling_rate)

    for ratio in ratios:
        chord = sum([chord, waveform(frequency * ratio[1] // ratio[0], sampling_rate)])

    return chord


def get_chord(chord: str) -> List[List[int]]:
    return [intervals[interval] for interval in chords[chord]]


def play_progression(emotion: str):
    progression = advanced_emotichords[emotion]

    for chord in progression:
        play_for(make_chord(notes['A'], get_chord(chord), sawtooth_wave), length)
