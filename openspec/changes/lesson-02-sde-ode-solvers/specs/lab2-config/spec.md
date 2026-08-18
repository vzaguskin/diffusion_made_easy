## ADDED Requirements

### Requirement: Solvers config section

В `configs/default.yaml` появляется секция `solvers` с полями: список методов (`euler_maruyama`, `euler`, `heun`, `rk4`), NFE-бюджет/число шагов, число сэмплов, seed. Все поля переопределяются из CLI как остальные секции лабы.

#### Scenario: CLI override of solver budget

- **WHEN** запущено `compare_solvers.py solvers.nfe_budget=100`
- **THEN** методы используют бюджет 100 (Euler — 100 шагов, Heun — 50, RK4 — 25), без правки YAML
