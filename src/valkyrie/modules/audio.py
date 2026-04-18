from __future__ import annotations

import numpy as np


class AudioEnhancementModule:
    name = "audio_enhancement"

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._noise_profile: np.ndarray | None = None

    def _estimate_noise(self, magnitude: np.ndarray) -> np.ndarray:
        # Use quietest 10% of frames as noise floor estimate
        frame_energy = np.mean(magnitude, axis=0)
        threshold = np.percentile(frame_energy, 10)
        noise_mask = frame_energy <= threshold
        if np.any(noise_mask):
            return np.mean(magnitude[:, noise_mask], axis=1, keepdims=True)
        return np.min(magnitude, axis=1, keepdims=True)

    def process(self, audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None or audio.size == 0:
            return audio

        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            return self._process_mono(audio)
        # Stereo: process each channel
        channels = [self._process_mono(audio[:, i]) for i in range(audio.shape[1])]
        return np.stack(channels, axis=1)

    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        n_fft = 2048
        hop = 512

        # STFT via numpy
        frames = []
        for start in range(0, len(audio) - n_fft, hop):
            frames.append(audio[start:start + n_fft] * np.hanning(n_fft))
        if not frames:
            return self._normalize(audio)

        stft = np.fft.rfft(np.array(frames), axis=1).T  # (freq, time)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise floor
        if self._noise_profile is None or self._noise_profile.shape[0] != magnitude.shape[0]:
            self._noise_profile = self._estimate_noise(magnitude)

        # Spectral subtraction with over-subtraction factor
        alpha = 2.0  # over-subtraction
        beta = 0.01  # spectral floor
        magnitude_clean = np.maximum(magnitude - alpha * self._noise_profile, beta * magnitude)

        # Update noise profile incrementally
        self._noise_profile = 0.95 * self._noise_profile + 0.05 * self._estimate_noise(magnitude)

        # Reconstruct STFT
        stft_clean = magnitude_clean * np.exp(1j * phase)

        # iSTFT via overlap-add
        result = np.zeros(len(audio), dtype=np.float32)
        window = np.hanning(n_fft)
        for i, start in enumerate(range(0, len(audio) - n_fft, hop)):
            if i >= stft_clean.shape[1]:
                break
            frame = np.fft.irfft(stft_clean[:, i], n=n_fft).real * window
            result[start:start + n_fft] += frame

        return self._normalize(result)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        # DC removal
        audio = audio - np.mean(audio)
        # Loudness normalization to -3 dBFS peak
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            target = 10 ** (-3.0 / 20.0)
            audio = audio * (target / peak)
        return audio.astype(np.float32)
