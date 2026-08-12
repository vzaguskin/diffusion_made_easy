## ADDED Requirements

### Requirement: Single default config file in YAML
Лаба SHALL иметь дефолтный конфиг `configs/default.yaml`, содержащий все гиперпараметры: параметры диффузии (`num_timesteps`, `beta_start`, `beta_end`), архитектуры (`model.type`, размеры), данных (`batch_size`, `num_workers`), оптимизатора (`type`, `lr`, `weight_decay`), обучения (`epochs`, `max_steps`), коллбэков (`sample_freq`, `num_samples`, `sampler`), путей (`data_dir`, `checkpoint_dir`, `log_dir`).

#### Scenario: Default config exists and is complete
- **WHEN** студент открывает `configs/default.yaml`
- **THEN** находит все перечисленные секции гиперпараметров с дефолтными значениями

### Requirement: Defaults fit 6GB VRAM
Дефолтные значения (`train.batch_size`, размеры U-Net, `num_samples`) SHALL быть подобраны так, чтобы запуск на GPU с 6GB VRAM на MNIST не падал с OOM без ручной настройки.

#### Scenario: Default run on 6GB GPU
- **WHEN** запускают `train.py` с дефолтным конфигом на 6GB GPU
- **THEN** обучение стартует и не падает с CUDA OOM

### Requirement: Config override via CLI
Скрипты `train.py` и `sample.py` SHALL позволять переопределять любые поля конфига через CLI в стиле `key=value` (через OmegaConf `from_cli`), поверх дефолтного файла. Должна быть возможность передать альтернативный файл конфига целиком.

#### Scenario: Override learning rate via CLI
- **WHEN** запускают `train.py optim.lr=5e-4`
- **THEN** обучение использует lr=5e-4 вместо дефолтного

#### Scenario: Custom config file
- **WHEN** запускают `train.py --config configs/small_model.yaml`
- **THEN** загружается указанный файл вместо `default.yaml`

### Requirement: Documented config schema in README
README SHALL документировать все ключи конфига с описанием и дефолтами, чтобы студент понимал, что каждый параметр делает.

#### Scenario: Student finds parameter explanation
- **WHЕН** студент видит в конфиге `num_timesteps: 1000`
- **THEN** в README он находит объяснение этого параметра и ссылку на раздел `theory.md`
