# Obliczanie MFCC z danego pliku mowy dla formatu wav.
import numpy as np
import matplotlib.pyplot as plt


# Konwersja z Hz do mel.
def hertz_to_mel(freq):
    return 1125 * np.log(1 + freq / 700)


# Konwersja z mel do Hz.
def mel_to_hertz(m):
    return 700 * (np.exp(m / 1125) - 1)
