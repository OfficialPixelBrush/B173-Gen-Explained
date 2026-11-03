import numpy as np

class JavaRandom:
    M_MULTIPLIER = np.uint64(0x5DEECE66D)
    M_ADDEND = np.uint64(0xB)
    M_MASK = np.uint64((1 << 48) - 1)

    def __init__(self, seed=0):
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = (
            np.uint64(seed) ^ self.M_MULTIPLIER
        ) & self.M_MASK

    def next(self, bits: int):
        self.seed = np.uint64(
            (int(self.seed) * int(self.M_MULTIPLIER)
            + int(self.M_ADDEND)) & int(self.M_MASK)
        )

        return np.int32(self.seed >> (48 - bits))

    def next_int(self):
        return self.next(32)

    def next_int_bound(self, bound):
        bound = np.int32(bound)

        if bound <= 0:
            raise ValueError("bound must be positive")

        if (bound & -bound) == bound:
            return np.int32(
                (np.int64(bound) * np.int64(self.next(31))) >> 31
            )

        while True:
            bits = self.next(31)
            val = bits % bound

            if np.int32(
                np.uint32(bits) - np.uint32(val)
                + np.uint32(bound - 1)
            ) >= 0:
                return val

    def next_long(self):
        return (
            (np.int64(self.next(32)) << 32)
            + np.int64(self.next(32))
        )

    def next_double(self):
        return np.float64(
            (np.int64(self.next(26)) << 27)
            + self.next(27)
        ) / np.float64(1 << 53)

    def next_boolean(self):
        return self.next(1) != 0

    def next_float(self):
        return np.float32(self.next(24)) / np.float32(1 << 24)
