"""Скрипт с исследованием решения задачи вахбы q-методом Давенпорта."""
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from environment.helpers import random_quaternion, generate_dirs, add_noise, quaternion_angle
from environment.quaternions import Quaternion


def q_method(reference: np.ndarray, observed: np.ndarray, weights: np.ndarray = None) -> Quaternion:
    """Решение задачи Вахбы через q-метод Давенпорта.

    Args:
        reference: Опорные векторы (N x 3)
        observed: Измеренные векторы (N x 3)
        weights: Веса измерений (N), если None - равные веса

    Returns:
        Кватернион поворота
    """
    if weights is None:
        weights = np.ones(len(reference))

    # Шаг 1: Создание матрицы B
    B = np.zeros((3, 3))
    for i in range(len(reference)):
        r = reference[i]  # эталонный вектор
        b = observed[i]  # измеренный вектор
        w = weights[i]  # вес

        # Внешнее произведение с весом
        B += w * np.outer(r, b)

    # Шаг 2: Создание вектора Z и матрицы Давенпорта K
    Z = np.array([
        B[1, 2] - B[2, 1],  # b23 - b32
        B[2, 0] - B[0, 2],  # b31 - b13
        B[0, 1] - B[1, 0]  # b12 - b21
    ])

    trace_B = np.trace(B)
    K = np.zeros((4, 4))
    K[0, 0] = trace_B
    K[0, 1:4] = Z
    K[1:4, 0] = Z
    K[1:4, 1:4] = B + B.T - np.eye(3) * trace_B

    # Шаг 3: Нахождение максимального собственного значения и вектора
    eigenvalues, eigenvectors = np.linalg.eig(K)
    max_idx = np.argmax(eigenvalues)
    M = eigenvectors[:, max_idx]

    # Шаг 4: Нормализация собственного вектора (это будет наш кватернион)
    q_normalized = M / np.linalg.norm(M)

    # Создаем кватернион (порядок: [w, x, y, z])
    return Quaternion(q_normalized)


def run_experiment(
    num_vectors: int = 10,
    noise_std: float = 0.01,
    num_trials: int = 1000,
) -> List[float]:
    """Проведение эксперимента по оценке точности q-метода.

    Args:
        num_vectors: Количество опорных направлений
        noise_std: СКО шума измерений
        num_trials: Количество испытаний

    Returns:
        angles: Список угловых ошибок в градусах
    """
    angles: List[float] = []

    for _ in range(num_trials):
        # Генерация случайного истинного вращения
        true_quat = random_quaternion()
        # Генерация опорных направлений
        reference = generate_dirs(num_vectors)
        # Применение вращения
        rotated = np.array([true_quat.rotate_vector(v) for v in reference])
        # Добавление шума
        observed = add_noise(rotated, noise_std)
        # Оценка вращения q-методом
        estimated_quat = q_method(reference, observed)
        # Вычисление ошибки
        angle_error = quaternion_angle(true_quat, estimated_quat)
        angles.append(angle_error)

    return angles


def plot_results(angles: List[float]):
    """Построение графиков распределения ошибок."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # График распределения угловых ошибок
    ax.hist(angles, bins=70, alpha=0.9, edgecolor='black')
    ax.set_xlabel('Угловая ошибка (градусы)')
    ax.set_ylabel('Частота')
    ax.set_title('Распределение угловых ошибок')
    ax.grid(True, alpha=0.3)

    # Добавление статистики
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    ax.axvline(
        mean_angle,
        color='red',
        linestyle='--',
        label=f'Среднее: {mean_angle:.2f}°',
    )
    ax.legend()

    plt.show()

    # Вывод статистики
    print(f"Статистика угловых ошибок:")
    print(f"  Среднее: {mean_angle:.4f}°")
    print(f"  СКО: {std_angle:.4f}°")
    print(f"  Медиана: {np.median(angles):.4f}°")
    print(f"  Максимум: {np.max(angles):.4f}°")
    print(f"  Минимум: {np.min(angles):.4f}°")


if __name__ == "__main__":
    # Параметры эксперимента
    np.random.seed(42)  # Для воспроизводимости

    angles = run_experiment(
        num_vectors=10,
        noise_std=0.001,  # Уровень шума 0.1%
        num_trials=2000,
    )

    plot_results(angles)
