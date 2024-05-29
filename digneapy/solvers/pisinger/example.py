from digneapy.solvers.pisinger import combo, minknap, expknap
import numpy as np


def main():
    n = 1000
    c = np.random.randint(1e3, 1e5)
    x = np.zeros(n, dtype=np.int32)
    w = np.random.randint(1000, 5000, size=n, dtype=np.int32)
    p = np.random.randint(1000, 5000, size=n, dtype=np.int32)
    minknap_time = minknap(n, p, w, x, c)
    combo_time = combo(n, p, w, x, c)
    expknap_time = expknap(n, p, w, x, c)
    print(
        f"MinKnap time {minknap_time}, Combo time: {combo_time}, ExpKnap time: {expknap_time}"
    )


if __name__ == "__main__":
    main()
