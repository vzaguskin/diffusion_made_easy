## 1. SDE-модуль (`src/score_lab/sde.py`)

- [x] 1.1 `ContinuousVE`: `f=0`, `g(t)=σ_min·(σ_max/σ_min)^t`; стартовое распределение `N(0, σ_max²I)`; отображение `t→idx` лестницы при вызове ε-модели
- [x] 1.2 `ContinuousVP`: `f=−½β(t)x`, `g=√β(t)` с линейной β; старт `N(0, I)`; `t→idx` по дискретным таймстепам
- [x] 1.3 Общий интерфейс: `eps_to_score`, `reverse_sde_drift(x,t)`, `pf_ode_drift(x,t)`, `diffusion(t)`, `prior_sample(shape, generator)`
- [x] 1.4 Тесты: `std(0)=0`, монотонность `std(t)`, `std(1)=σ_max`/`1`; согласие forward SDE с дискретным зашумлением (Euler-Maruyama vs `VESchedule`/`VPSchedule`)

## 2. Солверы (`src/score_lab/solvers.py`)

- [x] 2.1 Счётчик NFE как обёртка drift + wall-clock таймер (прогрев перед замером)
- [x] 2.2 `euler_maruyama(drift, diffusion, x, t_grid, generator)`: стохастический интегратор reverse SDE
- [x] 2.3 `ode_solver(method in {euler, heun, rk4}, drift, x, t_grid)`: детерминированные интеграторы PF-ODE (1/2/4 NFE на шаг)
- [x] 2.4 Преобразование NFE-бюджета в число шагов сетки для каждого метода (`nfe_to_grid`)
- [x] 2.5 Тесты: порядок точности (O(h)/O(h²)/O(h⁴)) на линейном ODE; NFE-счётчик равен фактическим вызовам; детерминизм ODE при повторном запуске; сходимость Euler-Maruyama на аналитическом гауссиане; согласие маргиналов SDE vs PF-ODE на гауссиане

## 3. Бенчмарк (`scripts/compare_solvers.py`)

- [x] 3.1 Загрузка чекпойнтов `model_ve.pt`/`model_vp.pt` с понятной ошибкой при отсутствии (+ подсказка `compare_ve_vp.py`)
- [x] 3.2 Матрица прогонов: {VE, VP} × {euler_maruyama, euler, heun, rk4} × NFE-бюджеты из конфига; один стартовый шум на ветку (seed из конфига); флаг `diverged` при NaN/±∞
- [x] 3.3 Метрики: sharpness, speckle, per-level ε-MSE сэмплов; запись в `solver_benchmark.csv` (метод, ветка, шаги, NFE, время, метрики)
- [x] 3.4 Сетки сэмплов по методам (PNG, общий заголовок с NFE) + график quality-vs-NFE (log-NFE, цвет=ветка, маркер=метод)
- [x] 3.5 Секция `solvers` в `configs/default.yaml` (методы, бюджеты, сэмплы, seed) + CLI-override

## 4. Документация и интеграция

- [x] 4.1 README лабы: раздел «SDE/ODE-солверы» — карта теории §11–15 → код, ожидания по времени, чтение quality-vs-NFE, ограничения метрик, упоминание DPM-Solver как next step
- [x] 4.2 Обновить «Регенерация картинок README» (прогон `compare_solvers.py`, копия в `../images/`)
- [x] 4.3 Полный прогон: тесты → `compare_solvers.py` на сохранённых чекпойнтах → копия артефактов в `images/` урока, коммит
