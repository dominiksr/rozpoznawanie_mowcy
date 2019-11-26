# Wektor kwantyzacji używając metody LBG. Tworzy codebook (słownik)
# odpowiednio grupując dane.
import numpy as np


# Obliczanie odległości euklidesowej pomiędzy dwiema macierzami.
def EUDistance(d, c):

    # Tworzenie zmiennych przechowywujących wymiar macierzy.
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n, p))
    if n < p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:, i], (p, 1)))
            distance[i, :] = np.sum((copies - c) ** 2, 0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:, i], (n, 1)))
            distance[:, i] = np.transpose(
                np.sum((d - copies) ** 2, 0)
            )  # RuntimeWarning;
    distance = np.sqrt(distance)

    return distance


# algorytm LBG.
def lbg(cechy, M):
    eps = 0.02
    codebook = np.mean(cechy, 1)
    distortion = 1
    nCentroid = 1
    while nCentroid < M:

        # Podwajanie wielkości codebook'a.
        nowy_codebook = np.empty((len(codebook), nCentroid * 2))
        if nCentroid == 1:
            nowy_codebook[:, 0] = codebook * (1 + eps)
            nowy_codebook[:, 1] = codebook * (1 - eps)
        else:
            for i in range(nCentroid):
                nowy_codebook[:, 2 * i] = codebook[:, i] * (1 + eps)
                nowy_codebook[:, 2 * i + 1] = codebook[:, i] * (1 - eps)

        codebook = nowy_codebook
        nCentroid = np.shape(codebook)[1]
        D = EUDistance(cechy, codebook)

        while np.abs(distortion) > eps:

            # Wyszukiwanie najbliższego sąsiada.
            poprzedni_distance = np.mean(D)
            najblizszy_codebook = np.argmin(D, axis=1)

            # Grupowanie wektorów i szukanie nowego centroidu.
            for i in range(nCentroid):

                # Dodawanie następuje w 3 wymiarze.
                codebook[:, i] = np.mean(
                    cechy[:, np.where(najblizszy_codebook == i)], 2
                ).T

            # Zamien wszystkie NaN na 0
            codebook = np.nan_to_num(codebook)
            # Pokaż 'this codebook', codebook
            D = EUDistance(cechy, codebook)
            distortion = (poprzedni_distance - np.mean(D)) / poprzedni_distance

    return codebook
