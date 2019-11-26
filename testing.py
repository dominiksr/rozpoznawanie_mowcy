# Testowanie mówców i wyświetlanie wyników.
import os
import numpy as np
from training import codebooks_mfcc, nSpeaker, nfiltbank
from LBG import EUDistance
from scipy.io.wavfile import read
from Metoda_MFCC import mfcc
from playsound import playsound

# Dane początkowe.
# Wskazanie katalogu gdzie przechowywane są pliki do testowania.
directory = os.getcwd() + "/test"
fname = str()
nCorrect_MFCC = 0

# Dopasowywanie trenowanego i testowanego mówcy.
def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        distance = EUDistance(features, codebooks[k, :, :])
        dist = np.sum(np.min(distance, axis=1)) / (np.shape(distance)[0])
        if dist < distmin:
            distmin = dist
            speaker = k

    return speaker


# Pętla wczytania pliku, testowania i pokazywania wyniku.
for i in range(nSpeaker):
    fname = "/s" + str(i + 1) + ".wav"
    print("Plik mówcy ", str(i + 1), "jest testowany.")
    (fs, s) = read(directory + fname)
    mel_coefs = mfcc(s, fs, nfiltbank)
    sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    print(
        "Mówca ", (i + 1), " pasuje do mówcy ", (sp_mfcc + 1),
    )
    if i == sp_mfcc:
        nCorrect_MFCC += 1

    # Odtwarzanie dźwięku.
    playsound(directory + fname)

# Wypisanie zgodności procentowej.
zgodn_proc_MFCC = (nCorrect_MFCC / nSpeaker) * 100
print("Zgodnoność procentowa dla wybranych plików:  ", zgodn_proc_MFCC, "%")
