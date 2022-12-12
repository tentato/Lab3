import time
import matplotlib.pyplot as plt
import numpy as np
import Z30 as z30

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
img = plt.imread("lenna.png")
img = img - np.min(img)
img /= np.max(img)
img *= 255
img = img.astype(int)

kernel = np.asarray([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
], dtype=int)

def main():
    start = time.perf_counter()
    conv_img = z30.seq_convolve(img, kernel)
    stop = time.perf_counter()
    print("Czas: ", stop - start)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(conv_img, cmap="gray")
    fig.savefig("Z30.png")

if __name__ == "__main__":
    main()