## ADDED Requirements

### Requirement: Loss bucketed by timestep intervals
Модуль метрик SHALL вычислять и логировать MSE-loss в разрезе по интервалам (basket'ам) `t` (например, 10 равных интервалов от 1 до `T`). Это иллюстрирует «отбрасывание веса» из раздела 15 теории: видно, как модель учится убирать шум разной интенсивности на слабо- vs сильно-зашумлённых шагах.

#### Scenario: Bucketed losses reported
- **WHEN** срабатывает валидация
- **THEN** в логе присутствуют значения loss для каждого интервала `t` (например, `val/loss_bucket_0` ... `val/loss_bucket_9`)

### Requirement: Coverage metric via nearest neighbors
Модуль метрик SHALL вычислять coverage: для партии сгенерированных картинок найти NN среди реальных картинок в пространстве пикселей (L2 по flatten) и посчитать долю уникальных реальных картинок, оказавшихся чьим-то NN. Значение близко к 1 = хорошее покрытие, близко к малому числу = mode-collapse. Пространство фич SHALL быть настраиваемо (дефолт — пиксели), README документирует возможность замены на CNN-фичи.

#### Scenario: Coverage reported on samples
- **WHEN** валидация генерирует партию сэмплов
- **THEN** в логе появляется скаляр `val/coverage` в диапазоне `[0, 1]`

### Requirement: Mode-collapse indicator via NN class distribution
Модуль метрик SHALL вычислять распределение классов (цифр 0-9) среди NN реальных картинок для сгенерированных сэмплов (используя метки тестового MNIST), и логировать это распределение (гистограмма по 10 классам). Если модель коллапсирует на少数 классы — гистограмма это покажет.

#### Scenario: Class distribution histogram logged
- **WHEN** срабатывает валидация с генерацией
- **THEN** в TensorBoard появляется гистограмма `val/nn_class_distribution` по 10 классам

### Requirement: Metrics avoid heavy external networks
Метрики SHALL НЕ требовать Inception-v3 или других тяжёлых внешних сетей. Все вычисления SHALL работать на пикселях или лёгких локальных операциях, чтобы сохранить учебную простоту и совместимость с MNIST.

#### Scenario: No Inception dependency
- **WHEN** прогоняют валидацию с метриками
- **THEN** ни одна метрика не загружает Inception или предобученную сеть
