## ADDED Requirements

### Requirement: True vs learned score quiver plot
Модуль визуализации SHALL строить парный quiver-график: истинный score (серый) и выученный `s_θ(x, σ)` (красный) на общей сетке точек, при заданном σ. Стрелки SHALL быть нормированы по длине (направление важнее магнитуды) с colorbar по логарифму нормы.

#### Scenario: Learned field visually close to truth
- **WHEN** сеть обучена до сходимости на среднем уровне лестницы
- **THEN** на quiver-графике направления красных и серых стрелок совпадают в high-density областях

### Requirement: Density heatmap with data samples
Модуль SHALL строить heatmap `log p(x)` (или `p_σ` на заданном уровне) с точками данных поверх.

#### Scenario: Heatmap follows the data
- **WHEN** рисуют плотность GMM поверх сэмплов датасета
- **THEN** максимум плотности проходит вдоль данных

### Requirement: Langevin trajectories plot
Модуль SHALL рисовать траектории Ланжевина (линии из стартовых точек, с маркером финальных позиций) поверх quiver-поля или heatmap. Число траекторий настраивается конфигом.

#### Scenario: Trajectories visible over the field
- **WHEN** рисуют 10–20 траекторий annealed Ланжевена
- **THEN** линии идут из шумного старта и стекают к модам, финальные точки — в high-density зонах

### Requirement: Annealing progress grid
Модуль SHALL строить сетку прогресса annealing: строки — уровни σ (от крупного к мелкому), колонки — шаги внутри уровня; каждая ячейка — scatter текущих позиций сэмплов. Это ключевой «видео-кадр» лабы: сгущение облака в моды по мере уменьшения σ.

#### Scenario: Cloud condenses across levels
- **WHEN** смотрят сетку annealing-прогресса
- **THEN** на верхних строках (большие σ) сэмплы размазаны широко, на нижних (малые σ) — собраны в узкие моды

### Requirement: Unified style and artifact saving
Все графики SHALL использовать единый стиль (данные — синие точки, истинный score — серый, выученный — красный) и сохраняться как PNG в каталог запуска (`runs/<exp>/`) с опциональным копированием в `lessons/02-score-matching/images/` для README.

#### Scenario: PNG written to run dir
- **WHEN** любой график создан
- **THEN** PNG-файл записан в `runs/<exp>/` с говорящим именем
