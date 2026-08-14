# noise-prediction-models Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: Models implement unified `forward(x, t) -> eps`
Каждая модель ε_θ SHALL реализовывать интерфейс `forward(x: Tensor[B,C,H,W], t: Tensor[B]) -> Tensor[B,C,H,W]`, где выход — предсказанный шум ε той же формы, что и вход. Это позволяет менять архитектуру одной настройкой конфига, не трогая ядро диффузии и сэмплеры.

#### Scenario: Output shape matches input
- **WHEN** в модель подают `x` формы `[64, 1, 28, 28]` и `t` формы `[64]`
- **THEN** выход имеет форму `[64, 1, 28, 28]`

#### Scenario: Both models are swappable
- **WHEN** конфиг переключает `model.type` с `mlp` на `unet`
- **THEN** остальной пайплайн (core, samplers, training) работает без изменений

### Requirement: Sinusoidal time embedding shared across models
Все модели SHALL обусловливаться на шаге `t` через общее sinusoidal positional time-embedding (стандарт для diffusion). Embedding-модуль SHALL быть единым для MLP и U-Net (`models/common.py`), чтобы не дублировать код.

#### Scenario: Same timestep gives same embedding
- **WHEN** две разные модели получают одинаковое `t`
- **THEN** их time-embedding (до проекции в размерности сети) идентичен

### Requirement: MLP option for transparency
Модуль `models/mlp.py` SHALL реализовывать ε_θ как MLP: flatten входа → time-embedding → concatenate/sum → residual MLP-блоки → reshape обратно в форму картинки. Назначение — максимальная прозрачность для учебы, не SOTA-качество.

#### Scenario: MLP is selectable via config
- **WHEN** конфиг задаёт `model.type: mlp`
- **THEN** фабрика моделей возвращает MLP-реализацию

### Requirement: U-Net option for quality
Модуль `models/unet.py` SHALL реализовывать ε_θ как U-Net с time-embedding, residual-блоками и up/down-sampling (классическая архитектура для diffusion). U-Net SHALL быть достаточно малым, чтобы вписываться в 6GB VRAM при дефолтном batch на MNIST.

#### Scenario: U-Net is the default
- **WHEN** конфиг не задаёт `model.type` явно
- **THEN** используется `unet` (дефолт)

#### Scenario: U-Net fits in 6GB VRAM
- **WHEN** U-Net обучается с дефолтным `train.batch_size` на MNIST на GPU 6GB
- **THEN** обучение не падает с CUDA OOM

### Requirement: Variance is not predicted by the network
Модели SHALL предсказывать только шум ε, не дисперсию `σ_t` (раздел 15 теории: дисперсия фиксируется, не обучается). Дисперсия для сэмплеров берётся из расписания шума (`β_t` или `~β_t`), а не из выхода сети.

#### Scenario: Model output is single tensor (eps only)
- **WHEN** модель вызывают на `(x, t)`
- **THEN** она возвращает один тензор (шум), а не кортеж `(eps, variance)`

