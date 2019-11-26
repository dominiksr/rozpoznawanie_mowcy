# Obliczanie MFCC z danego pliku mowy dla formatu wav.
import numpy as np
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct


# Konwersja z Hz do mel.
def hertz_to_mel(freq):
    return 1125 * np.log(1 + freq / 700)


# Konwersja z mel do Hz.
def mel_to_hertz(m):
    return 700 * (np.exp(m / 1125) - 1)


# Filtr częstotliwości mel.
# Częstotliwość podstawowa leży w zakresie 100-120 Hz w przypadku mężczyzn.
def mel_filterbank(nfft, nfiltbank, fs):
    gr_dolna = 100
    gr_gorna = 9000
    low_mel = hertz_to_mel(gr_dolna)
    upp_mel = hertz_to_mel(gr_gorna)

    # Zwraca liczbę równomiernie rozmieszczonych próbek,
    # obliczonych dla przedziału low_mel - upp_mel.
    mel = np.linspace(low_mel, upp_mel, nfiltbank + 2)

    # Powrotna zamiana na Hz.
    hertz = [mel_to_hertz(m) for m in mel]

    # Dostosowywanie częstotliwości.
    fbins = [int(hz * int(nfft / 2 + 1) / fs) for hz in hertz]
    fbank = np.empty((int(nfft / 2 + 1), nfiltbank))
    for i in range(1, nfiltbank + 1):
        for k in range(int(nfft / 2 + 1)):
            if k < fbins[i - 1]:
                fbank[k, i - 1] = 0
            elif k >= fbins[i - 1] and k < fbins[i]:
                fbank[k, i - 1] = (k - fbins[i - 1]) / (fbins[i] - fbins[i - 1])
            elif k >= fbins[i] and k <= fbins[i + 1]:
                fbank[k, i - 1] = (fbins[i + 1] - k) / (fbins[i + 1] - fbins[i])
            else:
                fbank[k, i - 1] = 0
    return fbank


def mfcc(s, fs, nfiltbank):
    # Podział na klatki 30 ms + 5 ms nakładu.
    klatka = 0.03
    naklad = 0.005
    nSamples = np.intc(klatka * fs)
    overlap = np.intc(naklad * fs)
    nFrames = np.intc(np.ceil(len(s) / (nSamples - overlap)))

    # Wypełnienie, by długość sygnału była wystarczająca,
    # aby uzyskać ilość ramek równą nFrame.
    wypelnienie = ((nSamples - overlap) * nFrames) - len(s)
    if wypelnienie > 0:
        signal = np.append(s, np.zeros(wypelnienie))
    else:
        signal = s

    # Nowa tablica o odpowiedniej pojemności, dla zapisania segmentu.
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:, i] = signal[start : start + nSamples]
        start = (nSamples - overlap) * i

    # Periodogram.
    nfft = 512  # 1024
    periodogram = np.empty((nFrames, int(nfft / 2 + 1)))
    for i in range(nFrames):
        # Użycie okna Hamminga. To stożek utworzony przy użyciu
        # podniesionego cosinusa z niezerowymi punktami końcowymi.
        x = segment[:, i] * hamming(nSamples)
        spectrum = fftshift(fft(x, nfft))
        periodogram[i, :] = abs(spectrum[int(nfft / 2 - 1) :]) / nSamples

    # Obliczanie MFCC.
    fbank = mel_filterbank(nfft, nfiltbank, fs)
    mel_coeff = np.empty((nfiltbank, nFrames))
    for i in range(nfiltbank):
        for k in range(nFrames):
            mel_coeff[i, k] = np.sum(periodogram[k, :] * fbank[:, i])

    # Dyskretna transformacja kosinusowa.
    mel_coeff = np.log10(mel_coeff)
    mel_coeff = dct(mel_coeff)

    # Wyklucz współczynnik zerowego rzędu.
    mel_coeff[0, :] = np.zeros(nFrames)

    return mel_coeff
