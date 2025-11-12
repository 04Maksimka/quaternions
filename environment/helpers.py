"""Всякие полезные функции."""
import numpy as np

from environment.quaternions import Quaternion


def add_noise(vectors: np.ndarray, noise_std: float) -> np.ndarray:
    """Добавление гауссова шума к векторам с последующей нормализацией."""
    noise = np.random.normal(0, noise_std, vectors.shape)
    noisy_vectors = vectors + noise
    return noisy_vectors / np.linalg.norm(noisy_vectors, axis=1, keepdims=True)


def random_quaternion() -> Quaternion:
    """Генерация случайного единичного кватерниона."""
    q = np.random.normal(0, 1, 4)
    return Quaternion(q / np.linalg.norm(q))


def quaternion_angle(q1: Quaternion, q2: Quaternion) -> float:
    """Вычисление угла между двумя кватернионами в градусах с учетом двойного покрытия."""
    dot = dot_prod_sqr(q1, q2)
    # Учитываем, что q и -q представляют одно вращение
    dot_abs = np.clip(np.abs(dot), 0.0, 1.0)
    angle_rad = 2 * np.arctan2(np.sqrt(1 - dot_abs**2), dot_abs)
    return np.degrees(angle_rad)

def generate_dirs(n: int):
    """Генерация n случайных единичных векторов в 3D."""
    vecs = np.random.randn(n, 3)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def dot_prod_sqr(q1: Quaternion, q2: Quaternion) -> float:
    return q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

def quat_metrics(q1: Quaternion, q2: Quaternion) -> float:
    return np.sqrt(dot_prod_sqr(q1 - q2, q1 - q2) * dot_prod_sqr(q1 + q2, q1 + q2))
