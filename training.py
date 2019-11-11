# Trening mówców i wyświetlanie wykresów.
import numpy as np
from scipy.io.wavfile import read
from Metoda_MFCC import mfcc
import matplotlib.pyplot as plt
import os


# Dane początkowe.
nSpeaker = 4
nCentroid = 16
nfiltbank = 12

# Tworzenie pustej przestrzeni.
codebooks_mfcc = np.empty((nSpeaker, nfiltbank, nCentroid))

# Wskazanie katalogu gdzie przechowywane są pliki do trenowania.
directory = os.getcwd() + "/train"

# Pętla wczytania pliku, trenowania i pokazywania wykresu.
for i in range(nSpeaker):
    fname = "/s" + str(i + 1) + ".wav"
    print("Plik mówcy ", str(i + 1), "jest trenowany.")

    # Wczytanie pliku.
    (fs, s) = read(directory + fname)

    # Zastosowanie metody trenującej.
    mel_coeff = mfcc(s, fs, nfiltbank)

    # Tworzenei wykresu.
    plt.figure(i)
    plt.title(
        "Codebook dla mócy " + str(i + 1) + " z " + str(nCentroid) + " centroidami"
    )
    plt.xlabel("Numer trenowanego mówcy")
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_ylabel("MFCC")

    for j in range(nCentroid):
        ax1.stem(codebooks_mfcc[i, :, j], use_line_collection=True)
