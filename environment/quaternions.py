import numpy as np


class Quaternion:
    """Класс кватерниона."""
    def __init__(self, w, x=None, y=None, z=None):
        # Если передан один аргумент (список/массив)
        if x is None and y is None and z is None:
            if isinstance(w, (list, np.ndarray)):
                if len(w) != 4:
                    raise ValueError("Для инициализации списком/массивом требуется 4 элемента")
                self.w, self.x, self.y, self.z = w
            else:
                raise TypeError("Для инициализации одним аргументом требуется список или массив")
        # Если переданы четыре числа
        else:
            self.w = w
            self.x = x
            self.y = y
            self.z = z

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            return Quaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            )
        elif isinstance(other, (int, float)):
            # Умножение каждого компонента на скаляр
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")

    def __rmul__(self, other):
        # Умножение скаляра на кватернион (коммутативно)
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Quaternion'")

    def __add__(self, other):
        if isinstance(other, Quaternion):
            # Покомпонентное сложение кватернионов
            return Quaternion(
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Quaternion' and '{type(other).__name__}'")

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            # Покомпонентное вычитание кватернионов
            return Quaternion(
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Quaternion' and '{type(other).__name__}'")

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self):
        norm = self.norm()
        return Quaternion(
            self.w / norm,
            self.x / norm,
            self.y / norm,
            self.z / norm
        )

    def as_rotation_matrix(self):
        w, x, y, z = self.normalized().w, self.x, self.y, self.z
        return np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

    def rotate_vector(self, vec):
        v_quat = Quaternion(0, vec[0], vec[1], vec[2])
        q_norm = self.normalized()
        result = q_norm * v_quat * q_norm.conjugate()
        return np.array([result.x, result.y, result.z])

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
