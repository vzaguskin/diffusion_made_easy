## ADDED Requirements

### Requirement: Default config with all sections
Лаба SHALL иметь `configs/default.yaml` со всеми секциями: `data` (dataset, n_samples, GMM-параметры), `model` (hidden_dim, layers, sigma-fourier features), `sigmas` (n_levels, sigma_max, sigma_min), `langevin` (steps_per_level, step_scale, n_trajectories), `train` (epochs, lr, batch_size, seed), `mnist_ve_vp` (ve-параметры, vp-параметры, epochs, model), `paths` (run_dir, images_dir).

#### Scenario: Complete default config
- **WHEN** студент открывает `configs/default.yaml`
- **THEN** находит все перечисленные секции с дефолтами и комментариями

### Requirement: CLI overrides via OmegaConf
Скрипты лабы SHALL поддерживать переопределение любых полей конфига через CLI `key=value` и альтернативный файл через `--config PATH` — тем же механизмом, что в лабе 1 (`src/score_lab/config.py`).

#### Scenario: Override dataset via CLI
- **WHEN** запускают `train_2d.py data.dataset=gaussians8`
- **THEN** обучение идёт на gaussians8 без правки файла конфига

### Requirement: Geometric sigma ladder helper
Конфиг-модуль (или core) SHALL предоставлять построение геометрической лестницы σ из `(sigma_max, sigma_min, n_levels)` — единую для обучения и Ланжевена.

#### Scenario: Ladder endpoints and length
- **WHEN** строят лестницу с `sigma_max=1.0, sigma_min=0.02, n_levels=10`
- **THEN** полученный тензор имеет длину 10, первый элемент 1.0, последний 0.02, промежуточные — геометрическая прогрессия
