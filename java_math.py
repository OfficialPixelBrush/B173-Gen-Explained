import numpy as np

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)
LIMIT_INT64 = 2.0**63

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
LIMIT_INT32 = 2.0**31


class JavaMath:
    PI = np.float64(3.141592653589793)
    PI_FLOAT = np.float32(PI)

    @staticmethod
    def lerp(t: np.float64, a: np.float64, b: np.float64):
        return a + t * (b - a)

    @staticmethod
    def _float_to_int(value, dtype, limit, min_value, max_value):
        arr = np.asarray(value)
        out = np.zeros(arr.shape, dtype=dtype)

        arr = arr.astype(np.float64)

        hi = arr >= limit
        lo = arr <= -limit
        valid = ~(hi | lo | np.isnan(arr))

        out[hi] = max_value
        out[lo] = min_value
        out[valid] = np.trunc(arr[valid]).astype(dtype)

        return out.item() if out.ndim == 0 else out

    @staticmethod
    def f64_to_int64(value):
        return JavaMath._float_to_int(
            value, np.int64, LIMIT_INT64, INT64_MIN, INT64_MAX
        )

    @staticmethod
    def f64_to_int32(value):
        return JavaMath._float_to_int(
            value, np.int32, LIMIT_INT32, INT32_MIN, INT32_MAX
        )

    @staticmethod
    def f32_to_int64(value):
        return JavaMath._float_to_int(
            value, np.int64, LIMIT_INT64, INT64_MIN, INT64_MAX
        )

    @staticmethod
    def f32_to_int32(value):
        return JavaMath._float_to_int(
            value, np.int32, LIMIT_INT32, INT32_MIN, INT32_MAX
        )

    @staticmethod
    def hash(value: str) -> np.int32:
        h = 0

        encoded = value.encode("utf-16-be", errors="surrogatepass")

        for i in range(0, len(encoded), 2):
            c = (encoded[i] << 8) | encoded[i + 1]
            h = (31 * h + c) & 0xFFFFFFFF

        if h >= 0x80000000:
            h -= 0x100000000

        return np.int32(h)


class MathHelper:
    TABLE_SIZE = 65536
    TABLE_MASK = TABLE_SIZE - 1
    RAD_TO_INDEX = 10430.378

    SIN_TABLE = np.sin(
        np.arange(TABLE_SIZE, dtype=np.float64)
        * JavaMath.PI * 2.0 / TABLE_SIZE
    ).astype(np.float32)

    @staticmethod
    def sin(value: np.float32) -> np.float32:
        index = JavaMath.f32_to_int32(
            np.float32(value * MathHelper.RAD_TO_INDEX)
        )
        return MathHelper.SIN_TABLE[int(index) & MathHelper.TABLE_MASK]

    @staticmethod
    def cos(value: np.float32) -> np.float32:
        index = JavaMath.f32_to_int32(
            np.float32(value * MathHelper.RAD_TO_INDEX + 16384.0)
        )
        return MathHelper.SIN_TABLE[int(index) & MathHelper.TABLE_MASK]

    @staticmethod
    def sqrt_f32(value: np.float32) -> np.float32:
        return np.float32(np.sqrt(value))

    @staticmethod
    def sqrt_f64(value: np.float64) -> np.float64:
        return np.float64(np.sqrt(value))

    @staticmethod
    def floor_f32(value: np.float32) -> np.int32:
        return np.int32(np.floor(value))

    @staticmethod
    def floor_f64(value: np.float64) -> np.int64:
        return np.int64(np.floor(value))

    @staticmethod
    def abs_f32(value: np.float32) -> np.float32:
        return np.abs(value)

    @staticmethod
    def abs_max_f64(a: np.float64, b: np.float64) -> np.float64:
        return np.maximum(np.abs(a), np.abs(b))