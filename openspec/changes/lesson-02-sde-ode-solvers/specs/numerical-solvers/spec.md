## Purpose

Численные интеграторы из §15 теории: Euler-Maruyama для стохастических SDE, Euler/Heun/RK4 для ODE — с общим интерфейсом, точным подсчётом NFE и замером wall-clock времени.

## ADDED Requirements

### Requirement: Euler-Maruyama for reverse SDEs

Солвер интегрирует reverse SDE по сетке времени: `x ← x + drift·Δt + g(t)·√Δt·z`, `z ~ N(0, I)`, с шагами, задаваемыми числом шагов из конфига. SHALL принимать генератор для воспроизводимости.

#### Scenario: Convergence on analytic Gaussian

- **WHEN** reverse SDE известного гауссиана интегрируется Euler-Maruyama с измельчающимся шагом
- **THEN** выборочные моменты конечного облака сходятся к аналитическим

### Requirement: Deterministic ODE solvers with a common interface

Для PF-ODE предоставляются методы Euler (1 NFE/шаг), Heun/RK2 (2 NFE/шаг) и RK4 (4 NFE/шаг) с одинаковыми сигнатурами: модель-дрейф, стартовое `x`, временная сетка, генератор (игнорируется ODE). Выбор метода — из конфига.

#### Scenario: Order of accuracy is observable

- **WHEN** линейный ODE с известным решением интегрируется каждым методом с уменьшающимся шагом
- **THEN** ошибка убывает как O(h) для Euler, O(h²) для Heun, O(h⁴) для RK4

#### Scenario: Same NFE budget, different step counts

- **WHEN** запрошен одинаковый NFE-бюджет для Euler и Heun
- **THEN** Heun использует вдвое меньшее число шагов временной сетки, и фактический NFE обоих запусков равен бюджету

### Requirement: NFE accounting and timing

Каждый запуск солвера SHALL возвращать: количество вычислений drift (NFE) — фактическое, а не номинальное; wall-clock время; флаг стохастичности. Счётчик NFE инкрементируется на каждый вызов модели.

#### Scenario: NFE counter matches model calls

- **WHEN** обёртка вокруг drift считает фактические вызовы
- **THEN** возвращённый NFE равен показанию обёртки для всех методов

#### Scenario: Deterministic solver returns zero randomness flag

- **WHEN** запущен ODE-солвер
- **THEN** повторный запуск с тем же стартом даёт бит-в-бит тот же результат при одинаковом NFE и времени ±шум таймера
