
## [Model Validation Samples](https://github.com/Nickel-nc/GU_Sport_DS/blob/master/2.%20Validation_Setup/2.%20Model_Validation_Samples.ipynb)

#### Contents:

* **EDA - Краткий обзор данных**
    - Графики распределения времени совершения транзакции в разрезе обучающей и тестовой выборок
    - Scatter-plot зависимости суммы транзакции от времени совершения транзакции
    - Графики зависимости даты транзакции в разрезе целевой переменной для обучающей и тестовой выборки
    - Распределение объемов транзакций (TransactionAmt) в логарифмическом масштабе
    - Распределение признака целевой переменной в зависимости от значений категориальных признаков ProductCD, card4, card6
    - Оценка пропусков
* **Hold-Out Validation (с разбиением на 2 выборки)**
    - xgb-model с подбором числа деревьев по early_stopping
* **Hold-Out Validation (с разбиением на 3 выборки)**
    - xgb-model с подбором числа деревьев по early_stopping
* **Доверительный интервал на основе BootStrap выборок**
    - Оценка доверительного интервала по Hold-Out Validation (с разбиением на 3 выборки)
* **Adversarial Validation**
    - Анализ гипотезы на train.csv / train + test.csv
* **KFold Validation**
    - Кросс-валидация на cross_val_score из пакета sklearn
    - KFold кросс-валидация на 5 фолдах. Общая оценка качества и разброса по метрике качества
    - KFold Validation по TimeSeriesSplit
* **Summaries**