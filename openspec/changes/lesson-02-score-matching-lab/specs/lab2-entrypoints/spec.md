## ADDED Requirements

### Requirement: train_2d.py runs the full 2D pipeline
Скрипт `scripts/train_2d.py` SHALL быть единой точкой входа 2D-части: загрузка конфига → обучение score-сети на лестнице σ → генерация всех визуализаций (quiver true-vs-learned, density, trajectories, annealing grid, single-σ collapse demo) → сохранение PNG в `runs/<exp>/` и CSV лоссов. Полный прогон на дефолтах SHALL занимать единицы минут.

#### Scenario: One command produces all artifacts
- **WHEN** выполняют `uv run python scripts/train_2d.py`
- **THEN** в `runs/<exp>/` появляются все типы графиков и CSV с кривой обучения

#### Scenario: Fast default run
- **WHEN** прогоняют дефолтный `train_2d.py`
- **THEN** завершение занимает не более ~10 минут на ноутбучном железе

### Requirement: compare_ve_vp.py runs the MNIST comparison
Скрипт `scripts/compare_ve_vp.py` SHALL обучать VE- и VP-модели последовательно с равным бюджетом, генерировать side-by-side сэмплы и график loss-кривых в `runs/ve_vs_vp/`.

#### Scenario: Comparison end-to-end
- **WHEN** выполняют `uv run python scripts/compare_ve_vp.py`
- **THEN** обе модели обучены, артефакты сравнения записаны, скрипт завершился без ошибок

### Requirement: README with expectations and theory map
Лаба SHALL иметь `README.md`: quickstart, карта «раздел теории → модуль кода», что ожидать от каждого графика (как читать quiver/annealing-grid), тюнинг Ланжевена (шаг), известные упрощения (GMM-аппроксимация, упрощённый VE-сэмплер), и как перегенерировать картинки README.

#### Scenario: Student can navigate code by theory section
- **WHEN** студент читает README раздел с картой теории
- **THEN** для каждого раздела §2/§4/§7/§8/§10 указан файл/функция, где он реализован

### Requirement: uv-managed reproducible environment
Лаба SHALL использовать uv (`pyproject.toml` + `uv.lock`); `uv sync` создаёт окружение; зависимости: torch, numpy, matplotlib, omegaconf, tqdm — без lightning/tensorboard.

#### Scenario: Sync and run on fresh machine
- **WHEN** на новой машине выполняют `uv sync` затем `uv run python scripts/train_2d.py`
- **THEN** пайплайн выполняется без ручной установки чего-либо

### Requirement: Unit tests for math-critical pieces
Лаба SHALL включать тесты: аналитический score GMM == autograd-градиент log p; DSM-цель с правильным знаком; Ланжевен с истинным score сходится к модам; mode-coverage annealed > single-σ.

#### Scenario: Test suite green
- **WHEN** запускают `uv run python -m pytest tests/ -q` (или run-вариант без pytest)
- **THEN** все проверки проходят
