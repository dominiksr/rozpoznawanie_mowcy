# Trening mówców i wyświetlanie wykresów.
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from LBG import lbg
from Metoda_MFCC import mfcc


# Dane początkowe.
nSpeaker = 7
nCentroid = 8  # Potęga 2.
nfiltbank = 10

# Tworzenie pustej przestrzeni.
codebooks_mfcc = np.empty((nSpeaker, nfiltbank, nCentroid))

# Wskazanie katalogu gdzie przechowywane są pliki do trenowania.
directory = os.getcwd() + "/train"
# fname = str()                                                  #Joanna BIN

# Pętla wczytania pliku, trenowania i pokazywania wykresu.
for i in range(nSpeaker):
    fname = "/s" + str(i + 1) + ".wav"
    print("Trenowany mówca nr: ", str(i + 1))

    # Wczytanie pliku.
    (fs, s) = read(directory + fname)

    # Zastosowanie metody trenującej.
    mel_coeff = mfcc(s, fs, nfiltbank)

    codebooks_mfcc[i, :, :] = lbg(mel_coeff, nCentroid)  ####Joanna changed

    # Tworzenei wykresu.
    plt.figure(i + 1)
    plt.title(
        "Codebook dla mówcy " + str(i + 1) + " z " + str(nCentroid) + " centroidami"
    )
    plt.xlabel("Liczba cech")
    ax1 = plt  # dla kilku subplot(2, 1, 1)
    plt.ylabel("MFCC")

    for j in range(nCentroid):
        ax1.stem(codebooks_mfcc[i, :, j], use_line_collection=True)

plt.show()
print("Trening zakończono. Przejście do dopasowywania mówców. ")
