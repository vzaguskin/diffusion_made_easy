## Purpose

Непрерывная SDE-постановка урока 2 (§11–14): forward SDE `dx = f(x,t)dt + g(t)dW` для VE и VP, reverse SDE и probability flow ODE, выраженные через одну обученную ε-модель. Даёт единый интерфейс дрейфа, поверх которого работают численные солверы.

## ADDED Requirements

### Requirement: Continuous VE and VP forward SDEs from the discrete schedules

Модуль предоставляет для каждой ветки (VE, VP) непрерывное описание прямого процесса в общей форме `dx = f(x, t)dt + g(t)dW` при `t ∈ [0, 1]`: VE — чисто аддитивный шум (`f = 0`, `g = σ(t)`); VP — линейный по β(t) drift с сохранением дисперсии. Функции согласованы с дискретными расписаниями лабы: интегрирование forward SDE от `t = 0` до `t` воспроизводит те же зашумления `x_t`, что и дискретные шаги (в пределах точности интегрирования).

#### Scenario: Forward SDE matches the discrete noising

- **WHEN** вычислено зашумление одного `x₀` дискретным расписанием (`VESchedule`/`VPSchedule`) и Euler-Maruyama-интегрированием forward SDE до того же уровня шума
- **THEN** средняя разница по большому числу повторений стремится к нулю с измельчением шага

#### Scenario: Shared time parameterization

- **WHEN** запрошен уровень шума `std(t)` для любого `t ∈ [0, 1]`
- **THEN** `std` монотонно не убывает, у VE `std(0) = σ_min ≈ 0` и `std(1) = σ_max`, у VP `std(0) ≈ 0` и `std(1) = 1`

### Requirement: Reverse SDE drift from an eps-model

Для каждой ветки reverse SDE `dx = [f(x,t) − g²(t)·s_θ(x,t)]dt + g(t)dW̄` вычисляется через score `s_θ = −ε_θ/std(t)` из той же ε-модели, что использовалась в дискретной части лабы. Преобразование ε ↔ score SHALL использовать формулы §9: VE — `−ε/σ`, VP — `−ε/√(1−ᾱ)`.

#### Scenario: Score conversion per branch

- **WHEN** ε-модель вернула `ε_θ(x, t)` для обеих веток
- **THEN** reverse-SDE drift использует score, делённый на std шума соответствующей ветки, без дублирования сетей или весов

### Requirement: Probability flow ODE drift

PF-ODE `dx/dt = f(x,t) − ½·g²(t)·s_θ(x,t)` предоставляется для обеих веток через тот же score и тот же интерфейс `drift(x, t)`, что и reverse SDE. Решения PF-ODE имеют те же маргинальные распределения, что и reverse SDE (проверяемо на аналитическом случае).

#### Scenario: ODE and SDE marginals agree on Gaussian data

- **WHEN** на одномерном/двумерном гауссиане с аналитическим score запущены reverse SDE и PF-ODE от одного стартового распределения
- **THEN** выборочные среднее и ковариация конечных точек совпадают в пределах выборочной погрешности

#### Scenario: Deterministic given the start

- **WHEN** PF-ODE интегрируется дважды с одним стартом и фиксированной моделью
- **THEN** траектории и конечные точки идентичны (ODE не содержит случайности)
