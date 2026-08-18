## ADDED Requirements

### Requirement: compare_solvers.py runs the SDE/ODE benchmark

Скрипт запускает бенчмарк солверов (см. capability `solver-benchmark`) на чекпойнтах из `runs/ve_vs_vp/`, без повторного обучения. Поддерживает CLI-override конфига как остальные скрипты лабы (`key=value`). Ожидаемое время прогона указано в README.

#### Scenario: End-to-end run

- **WHEN** выполнен `uv run python scripts/compare_solvers.py` при наличии чекпойнтов
- **THEN** в `runs/ve_vs_vp/solvers/` появляются сетки сэмплов по методам, `solver_benchmark.csv` и `quality_vs_nfe.png`

### Requirement: README covers the SDE/ODE section

README лабы содержит: раздел «SDE/ODE-солверы» с картой «раздел теории → код» для §11–15, ожидания по времени прогона, как читать график quality-vs-NFE, и ограничения метрик качества. Существующие разделы (2D, VE vs VP, известные упрощения) обновлены ссылками на новую часть.

#### Scenario: Theory map updated

- **WHEN** читается таблица «раздел теории → код» в README
- **THEN** в ней есть строки для §11–15 (Wiener, forward SDE, reverse SDE, PF-ODE, солверы) с указанием модулей `sde.py`/`solvers.py`/`compare_solvers.py`
