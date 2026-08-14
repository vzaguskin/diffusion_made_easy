## ADDED Requirements

### Requirement: Lesson labs are self-contained
Лаба каждого урока SHALL быть самодостаточной: иметь собственный `pyproject.toml`/`uv.lock` и НЕ импортировать код лаб других уроков. Копирование кода между лабами допустимо (с адаптацией и комментарием), если это делает урок независимым от прохождения предыдущих. Мотивация: уроки можно проходить и запускать в любом порядке.

#### Scenario: Lesson 2 lab does not import lesson 1 code
- **WHEN** смотрят импорты в `lessons/02-score-matching/lab/src/`
- **THEN** ни один импорт не ссылается на `lessons/01-*` или `ddpm_lab`

#### Scenario: Lab runs without other lessons present
- **WHEN** каталог `lessons/02-score-matching/lab/` копируют в изолированное место и запускают `uv sync` + скрипты
- **THEN** всё работает без доступа к другим урокам
