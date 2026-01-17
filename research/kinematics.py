"""Скрипт исследования численного решения кинематического уравнения.

Постановка: есть датчик угловой скорости (ДУС) и мы снимаем его показания
в связной системе координат (ССК) у нас также есть q_ИСК_ССК,
ИСК -- инерциальная СК, нужно реализовать вычислительно простой и достаточно
точный метод численного интегрирования. Простота нужна так как в
космосе мало вычислительного ресурса.
"""
import numpy as np
from matplotlib import pyplot as plt

from environment.helpers import quaternion_angle, quat_metrics
from environment.quaternions import Quaternion


def exact_kinematic_solution(
        q_initial: Quaternion,
        omega: np.array,
        dt: float,
) -> Quaternion:
    """Точное решение для постоянной угловой скорости.

    :param q_initial: Начальный кватернион ИСК-ССК
    :param omega: угловая скорость в ССК
    :param dt: величина времени движения
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return q_initial
    n = omega / omega_norm
    theta = omega_norm * dt

    q_delta = Quaternion(
        np.cos(theta / 2),
        *(n * np.sin(theta / 2))
    )
    return (q_initial * q_delta).normalized()


def r2_deg_solution(
        q_initial: Quaternion,
        omega: np.array,
        dt: float,
) -> Quaternion:
    """Численное решение методом разложения в ряд Тейлора до 2-го порядка."""
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return q_initial

    n = omega / omega_norm
    theta = omega_norm * dt

    # Используем половинный угол φ = theta/2
    phi = theta / 2

    q_delta = Quaternion(
        1 - phi ** 2 / 2 - phi ** 4 / 32,
        *(n * (phi - phi ** 5 / 8))
    )
    return q_initial * q_delta


def r2_deg_modified_solution(
        q_initial: Quaternion,
        omega: np.array,
        dt: float,
) -> Quaternion:
    """Численное решение методом разложения модифицированный."""
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return q_initial

    n = omega / omega_norm
    theta = omega_norm * dt

    # Используем половинный угол φ = theta/2
    phi = theta / 2

    q_delta = Quaternion(
        1 - phi ** 2 / 2 - phi ** 4 / 24,
        *(n * (phi - phi ** 3 / 6 - phi ** 5 / 72))
    )
    return q_initial * q_delta


def get_omega(step: int, dt: float) -> np.array:
    """Генератор угловой скорости, зависящей от шага и времени.

    :param step: номер шага
    :param dt: шаг времени
    :return: вектор угловой скорости [рад/с]
    """
    t = step * dt
    # Пример: изменяющаяся угловая скорость
    # Можно настроить по необходимости
    base_omega = 0.1  # базовая угловая скорость

    # Добавляем гармонические изменения по разным осям
    omega_x = base_omega * (1 + 0.5 * np.sin(0.1 * t))
    omega_y = base_omega * (1 + 0.3 * np.cos(0.15 * t + 0.5))
    omega_z = base_omega * (1 + 0.2 * np.sin(0.2 * t + 1.0))

    return np.array([omega_x, omega_y, omega_z])


def main():
    """Эксперимент."""
    # Начальные условия
    q0 = Quaternion(0.5, 0.5, 0.5, 0.5)
    dt = 0.1  # Шаг интегрирования [с]
    steps = 1_000  # Число шагов

    # Траектории решений
    q_exact = q0
    q_r2deg = q0
    q_r2deg_modified = q0
    errors_r2deg = []
    errors_r2deg_modified = []
    error_ratio = []  # Отношение ошибок
    omega_history = []  # История угловых скоростей

    for step in range(steps):
        # Получаем угловую скорость для текущего шага
        omega = get_omega(step, dt)
        omega_history.append(omega.copy())

        q_exact = exact_kinematic_solution(q_exact, omega, dt)
        q_r2deg = r2_deg_solution(q_r2deg, omega, dt)
        q_r2deg_modified = r2_deg_modified_solution(q_r2deg_modified, omega, dt)

        error_r2deg = quat_metrics(q_exact, q_r2deg)
        error_r2deg_modified = quat_metrics(q_exact, q_r2deg_modified)

        errors_r2deg.append(error_r2deg)
        errors_r2deg_modified.append(error_r2deg_modified)
        if error_r2deg_modified > 1e-12:
            error_ratio.append(error_r2deg / error_r2deg_modified)
        else:
            error_ratio.append(1.0)

    # Преобразуем историю в numpy array для удобства
    omega_history = np.array(omega_history)

    # Создаем фигуру с тремя подграфиками
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # Левый график - ошибки методов
    ax1.plot(errors_r2deg, label='Разложение в ряд Тейлора')
    ax1.plot(errors_r2deg_modified, label='Модифицированное разложение')
    ax1.set_title("Сравнение методов численного интегрирования")
    ax1.set_ylabel("Ошибка, |q - p||q + p|")
    ax1.set_xlabel(f"Шаг интегрирования (время × {dt} с)")
    ax1.grid(True)
    ax1.legend()

    # Средний график - отношение ошибок
    ax2.plot(error_ratio, color='red', linewidth=2)
    ax2.set_title(f"Отношение ошибок, сред. знач. = {np.mean(error_ratio):.1f}")
    ax2.set_xlabel(f"Шаг интегрирования (время × {dt} с)")
    ax2.grid(True)

    # Правый график - компоненты угловой скорости
    time = np.arange(steps) * dt
    ax3.plot(time, omega_history[:, 0], label='ω_x')
    ax3.plot(time, omega_history[:, 1], label='ω_y')
    ax3.plot(time, omega_history[:, 2], label='ω_z')
    ax3.set_title("Компоненты угловой скорости")
    ax3.set_xlabel("Время, с")
    ax3.set_ylabel("Угловая скорость, рад/с")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()