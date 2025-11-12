"""Исследование функции усреднения от n кватернионов.

Так же как и задача Вахбы среднее от кватернионов
приводят к задаче argmax Tr(A * B), где A - известная матрица,
а B -- искомая ортогональная матрица.
"""
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from environment.helpers import quaternion_angle, add_noise, random_quaternion, generate_dirs
from environment.quaternions import Quaternion
from research.wahba_problem import q_method


def calc_average_quaternion(
        quats: List[Quaternion],
        weights: List[float] | None = None,
) -> Quaternion:
    if weights is None:
        weights = np.full_like(quats, 1)
    if len(quats) != len(weights):
        raise ValueError("Количество кватернионов и весов должно совпадать")
    if len(quats) == 0:
        raise ValueError("Список кватернионов не может быть пустым")

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Сумма весов должна быть положительной")
    normalized_weights = [w / total_weight for w in weights]

    # Построение матрицы M
    M = np.zeros((4, 4))
    for q, w in zip(quats, normalized_weights):
        # Преобразование кватерниона в вектор numpy
        q_vec = np.array([q.w, q.x, q.y, q.z])
        # Внешнее произведение: q_vec * q_vec^T
        M += w * np.outer(q_vec, q_vec)

    # Вычисление собственных значений и векторов
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Нахождение индекса максимального собственного значения
    max_eigenvalue_index = np.argmax(eigenvalues)
    # Извлечение соответствующего собственного вектора
    average_q_vec = eigenvectors[:, max_eigenvalue_index]
     # Нормализация собственного вектора (должна быть единичной, но для точности делаем)
    average_q_vec = average_q_vec / np.linalg.norm(average_q_vec)

    return Quaternion(average_q_vec)


def averaging_experiment(
        num_quats: int,
        num_vectors=10,
        noise_std=0.001,
        num_trials=2000,
):
    """
    Эксперимент по исследованию среднего кватерниона.

    Args:
        num_quats: количество звездных датчиков
        num_vectors: количество опорных направлений для каждого датчика
        noise_std: СКО шума измерений
        num_trials: количество испытаний
    """
    single_errors = []
    averaged_errors = []

    for _ in range(num_trials):
        # Генерация случайного истинного вращения
        true_quat = random_quaternion()

        # Каждый датчик решает свою задачу Вахбы с независимыми измерениями
        measured_quats = []

        for _ in range(num_quats):
            # Генерация опорных направлений (разные для каждого датчика)
            reference = generate_dirs(num_vectors)
            # Применение истинного вращения
            rotated = np.array([true_quat.rotate_vector(v) for v in reference])
            # Добавление независимого шума
            observed = add_noise(rotated, noise_std)
            # Решение задачи Вахбы для данного датчика
            estimated_quat = q_method(reference, observed)
            measured_quats.append(estimated_quat)

        # Ошибка одного (первого) датчика
        single_error = quaternion_angle(true_quat, measured_quats[0])
        single_errors.append(single_error)

        # Усреднение показаний всех датчиков
        averaged_quat = calc_average_quaternion(measured_quats)
        averaged_error = quaternion_angle(true_quat, averaged_quat)
        averaged_errors.append(averaged_error)

    return single_errors, averaged_errors


def plot_averaging_results(single_errors, averaged_errors, num_quats: int):
    """Построение графиков сравнения ошибок."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Гистограмма распределения ошибок
    bins = np.linspace(
        0,
        max(max(single_errors),
        max(averaged_errors)) * 1.1,
        50,
    )

    ax1.hist(
        single_errors,
        bins=bins,
        alpha=0.5,
        label='Один датчик',
        color='blue',
        edgecolor='black',
    )
    ax1.hist(
        averaged_errors,
        bins=bins,
        alpha=0.5,
        label=f'Усреднение {num_quats} датчиков',
        color='red',
        edgecolor='black',
    )
    ax1.set_xlabel('Угловая ошибка (градусы)')
    ax1.set_ylabel('Частота')
    ax1.set_title('Сравнение распределения угловых ошибок')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot для наглядного сравнения
    data = [single_errors, averaged_errors]
    labels = ['Один датчик', f'Усреднение\n{num_quats} датчиков']

    ax2.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    ax2.set_ylabel('Угловая ошибка (градусы)')
    ax2.set_title('Box plot угловых ошибок')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("=" * 50)
    print(f"СТАТИСТИКА ДЛЯ {num_quats} ДАТЧИКОВ")
    print("=" * 50)
    print("Один датчик:")
    print(f"  Среднее: {np.mean(single_errors):.6f}°")
    print(f"  СКО: {np.std(single_errors):.6f}°")
    print(f"  Медиана: {np.median(single_errors):.6f}°")
    print(f"  95% перцентиль: {np.percentile(single_errors, 95):.6f}°")

    print(f"\nУсреднение {num_quats} датчиков:")
    print(f"  Среднее: {np.mean(averaged_errors):.6f}°")
    print(f"  СКО: {np.std(averaged_errors):.6f}°")
    print(f"  Медиана: {np.median(averaged_errors):.6f}°")
    print(f"  95% перцентиль: {np.percentile(averaged_errors, 95):.6f}°")

    improvement_mean = (1 - np.mean(averaged_errors) / np.mean(single_errors)) * 100
    improvement_std = (1 - np.std(averaged_errors) / np.std(single_errors)) * 100
    print(f"\nУлучшение за счет усреднения:")
    print(f"  По среднему: {improvement_mean:.2f}%")
    print(f"  По СКО: {improvement_std:.2f}%")


if __name__ == "__main__":
    np.random.seed(42)
    num_sensors = 4

    print(f"\n{'=' * 60}")
    print(f"ЭКСПЕРИМЕНТ С {num_sensors} ДАТЧИКАМИ")
    print(f"{'=' * 60}")

    single, averaged = averaging_experiment(
        num_quats=num_sensors,
        num_vectors=10,
        noise_std=0.001,
        num_trials=1000
        )

    plot_averaging_results(single, averaged, num_sensors)