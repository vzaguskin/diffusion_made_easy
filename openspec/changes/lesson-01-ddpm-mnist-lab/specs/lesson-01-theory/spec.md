## ADDED Requirements

### Requirement: Lesson 1 theory covers full DDPM derivation
Файл `lessons/01-ddpm-from-scratch/theory.md` SHALL содержать полный вывод DDPM, включая как минимум следующие разделы: идея одним абзацем; прямой процесс (forward) с variance-preserving правилом; трюк прыжка через все шаги (`x_t = √ᾱ·x₀ + √(1-ᾱ)·ε`); ELBO и неравенство Йенсена; разложение ELBO на KL-члены; формула Байеса для заднего распределения; марковость; произведение гауссианов и выделение полного квадрата; репараметризация через шум; финальная потеря `L = E[‖ε − ε_θ(x_t,t)‖²]`; алгоритм обучения; алгоритм генерации (DDPM); DDIM.

#### Scenario: All key derivation stages present
- **WHEN** студент открывает `theory.md` урока 1
- **THEN** он находит разделы для каждого из перечисленных этапов вывода, с формулами

### Requirement: Final loss formula stated explicitly
`theory.md` SHALL явно фиксировать финальную функцию потерь `L_DDPM = E_{t, x₀, ε}[‖ε − ε_θ(x_t, t)‖²]` и эмпирический трюк «отбрасывания веса» (приравнивание весового коэффициента к 1 для всех `t`).

#### Scenario: Loss formula and weight-dropping trick
- **WHEN** студент читает раздел про финальную потерю
- **THEN** видит формулу `L = E[‖ε − ε_θ‖²]` и объяснение, почему вес при `t` приравнен к 1

### Requirement: Training and inference algorithms stated as pseudocode
`theory.md` SHALL содержать алгоритм обучения DDPM (шаги: x₀ → t → ε → x_t → ŝ = ε_θ(x_t,t) → loss = ‖ε − ŝ‖²) и алгоритм генерации (цикл t = T...1 с обновлением `x_{t-1} = μ + σ_t·z`, без шума на последнем шаге) в виде псевдокода.

#### Scenario: Training algorithm
- **WHEN** студент читает раздел про алгоритм обучения
- **THEN** видит пошаговый псевдокод цикла обучения

#### Scenario: Inference algorithm
- **WHEN** студент читает раздел про алгоритм генерации
- **THEN** видит цикл денуазинга и условие «на последнем шаге шум не добавляется»
