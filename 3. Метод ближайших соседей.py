# ---------------------------------------------------------
# Машинное обучение для экономистов
# Тема - Метод ближайших соседей
# ---------------------------------------------------------

# ---------------------------------------------------------
# Установка бибилотек (в командной строке)
# ---------------------------------------------------------
# python.exe -m pip install --upgrade pip
# pip install numpy
# pip install pandas
# pip install scikit-learn
# pip install bnlearn
# pip install openpyxl
# pip install d3blocks
# ---------------------------------------------------------

# Подключим необходимые библиотеки
import numpy as np                                        # базовые операции с массивами
import pandas as pd                                       # базовые операции с датафреймами
from sklearn import tree                                  # решающие деревья
from sklearn.ensemble import RandomForestClassifier       # случайный лес
from sklearn.model_selection import train_test_split      # разделение выборки на 
from sklearn.model_selection import cross_val_score       # кросс-валидация
from sklearn.model_selection import KFold                 # разбиение на фолды
from sklearn.utils import shuffle                         # случайная перестановка
from sklearn.inspection import permutation_importance     # перестановочная важность
import openpyxl                                           # работа с excel
import d3blocks                                           # интерактивные графики
import random                                             # случайные числа 
import matplotlib.pyplot as plt                           # графики
from sklearn.metrics import confusion_matrix              # подсчет прогнозов
from sklearn.neighbors import KNeighborsClassifier        # ближайшие соседи
import sklearn
from sklearn.model_selection import GridSearchCV          # подбор гиперпараметров
from sklearn.model_selection import RandomizedSearchCV    # подбор гиперпараметров
import scipy
from sklearn.metrics import RocCurveDisplay               # ROC-кривая
from sklearn import metrics                               # метрики
import itertools
import kds                                                # выигрыш (gain)

# Рассматриваемые методы
# 1. Метод ближайших соседей

# Рассматриваемые техники сравнения качества моделей
# 1. ROC-кривая
# 2. Выигрыш (gain)

# ---------------------------------------------------------
# Загрузка и первичный анализ данных
# ---------------------------------------------------------

# Загрузим данные
df = pd.read_excel("E:\Преподавание\Машинное обучение\Данные\Продажи.xlsx")

# Посмотрим первые несколько строк в данных
df.head(5)

# Сохраним число наблюдений
n = df.index.size

# Разделим целевую переменную и признаки
target = df.loc[:, ['buy']]                     # целевая переменная
features = df.loc[:, df.columns.drop('buy')]    # матрица признаков
target = np.squeeze(target)                     # преобразуем из вектора столбца
                                                # в одномерный массив

# ---------------------------------------------------------
# Метод блилайжих соседей
# ---------------------------------------------------------

# Нормализуем данные
scaler = sklearn.preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
# Попробуйте самостоятельно сравнить с результатами без нормализации

# Воспользуемся методом ближайших соседей
knn = KNeighborsClassifier(n_neighbors = 3,      # число соседей
                           metric = "minkowski", # метрика расстояния
                           p = 2)                # Евклидова метрика
knn.fit(features, target)

# Оцениваем вероятности
prob_knn = knn.predict_proba(features)
prob_knn[0:10:, 0]                               # оценки P(Y = 0 | X = x)
prob_knn[0:10:, 1]                               # оценки P(Y = 1 | X = x)

# Предсказания
prediction_knn = knn.predict(features)           # I(P(Y = 1 | X = x) > 0.5)

# Оценим точность
ACC_knn = knn.score(features, target)            

# ---------------------------------------------------------
# Тюнинг гиперпараметров на примере подбора оптимального
# числа соседей и метрики расстояния
# ---------------------------------------------------------

# Подробно о тюнинге параметров
# https://scikit-learn.org/stable/modules/grid_search.html

# Гиперпараметры модели
knn.get_params()

# Перебираемые значения гиперпараметров
hyperparameters = {'n_neighbors': [3, 5, 7], 'p': [1, 2, 3]}

# Перебор гиперпараметров с помощью кросс-валидации
GSCV_knn = GridSearchCV(estimator = knn, 
                        param_grid = hyperparameters, 
                        scoring = "accuracy",
                        cv = 5)
GSCV_knn.fit(features, target)

# Достанем гиперпараметры, соответствующие лучшей модели
hyperparameters_best = GSCV_knn.best_params_

# Оценим модель с лучшими параметрами
knn2 = KNeighborsClassifier(n_neighbors = hyperparameters_best["n_neighbors"],
                            metric = "minkowski",
                            p = hyperparameters_best["p"])
knn2.fit(features, target)

# Подбор гиперпараметров с помощью рандомизированного поиска
hyperparameters2 = dict(n_neighbors = scipy.stats.poisson(mu = 2, loc = 1),
                        p = scipy.stats.geom(p = 0.5, loc = 1))
RPCV_knn = RandomizedSearchCV(estimator = knn, 
                              param_distributions = hyperparameters2, 
                              scoring = "accuracy",
                              cv = 5,
                              n_iter = 10)          # число симуляций
RPCV_knn.fit(features, target)
hyperparameters2_best = RPCV_knn.best_params_

# Сравним результаты двух подходов
print([GSCV_knn.best_score_, RPCV_knn.best_score_])

# ---------------------------------------------------------
# ROC-кривая
# ---------------------------------------------------------

# Разделим выборку на тестовую и тренеровочную
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.2, random_state = 777)
    
# Оценим модели на обучающей выборке
knn.fit(features_train, target_train)
knn2.fit(features_train, target_train)
    
# Построим ROC-кривую на тестовой выборке
RocCurveDisplay.from_estimator(knn, features_test, target_test)
plt.show()
plt.close()

# Оценим вероятности на тестовой выборке
knn_prob1 = knn.predict_proba(features_test)[:, 1]
knn2_prob1 = knn2.predict_proba(features_test)[:, 1]

# Посчитеаем TRP и FPR при различных порога (thresholds)
knn_FPR, knn_TRP, knn_thresholds = metrics.roc_curve(target_test, knn_prob1)
knn2_FPR, knn2_TRP, knn2_thresholds = metrics.roc_curve(target_test, knn2_prob1)
  
# Построим ROC-кривые сразу для двух моделей
plt.plot(knn_FPR, knn_TRP, label = "knn")
plt.plot(knn2_FPR, knn2_TRP, label = "knn2")
plt.show()
plt.close()

# Сравним модели по AUC
knn_AUC = metrics.roc_auc_score(target_test, knn_prob1)
knn2_AUC = metrics.roc_auc_score(target_test, knn2_prob1)
print([knn_AUC, knn2_AUC])

# ---------------------------------------------------------
# Оценивание выигрыша (gain)
# ---------------------------------------------------------

# Воспользуемся методом ближайших соседей с весами
knn3 = KNeighborsClassifier(n_neighbors = 99,      # число соседей
                            metric = "minkowski",  # метрика расстояния
                            p = 2,                 # Евклидова метрика
                            weights = "distance")  # веса                
knn3.fit(features_train, target_train)

# Создадим набор данных из значений целевой переменной на тестовой
# выборке и соответствующих оценок условных вероятностей
df_gain = pd.DataFrame() 
df_gain["target"] = target_test
df_gain["prob"] = knn3.predict_proba(features_test)[:, 1]

# Отсортируем датафрейм по вероятностям
df_gain = df_gain.sort_values(by = "prob", ascending = False)

# Разделим датафрейм на децили
df_gain['decile'] = pd.qcut(df_gain['prob'], q = 10, labels = range(10, 0, -1))

# Создадим таблицу для сохранения результатов расчета выигрышей
tbl_gain = pd.DataFrame()

# Создадим столбец для децилей
tbl_gain["decile"] = range(1, 11)

# Посчитаем число наблюдений в каждой децили
tbl_gain["obs"] = np.array(df_gain.groupby(
  'decile')['target'].count().sort_index(ascending = False))

# Рассчитаем число 1, расположенных в соответствующих децилях
tbl_gain["ones"] = np.array(df_gain.groupby(
  'decile')['target'].sum().sort_index(ascending = False))
  
# Посчитаем выигрыш
n_ones = np.sum(tbl_gain["ones"])
tbl_gain["gain"] = 100 * tbl_gain["ones"] / n_ones

# Вычислим кумулятивный выигрыш
tbl_gain["cumulative_gain"] = tbl_gain["gain"].cumsum()

# Вычислим лифт
ones_per_decile_random = n_ones / len(tbl_gain)
tbl_gain["lift"] = tbl_gain["ones"] / ones_per_decile_random

# Посчитаем кумулятивный лифт
cum_ones_per_decile_random = np.repeat(
  ones_per_decile_random, len(tbl_gain)).cumsum()
tbl_gain["cumulative_lift"] = cum_model / cum_ones_per_decile_random

# Предположим, что затраты на привлечение потенциального покупателя
# составляют 1 рубль, а доход с продажи 10 рублей
p_revenue = 10
p_cost = 1
tbl_gain["revenue"] = tbl_gain["ones"] * p_revenue
tbl_gain["cost"] = tbl_gain["obs"] * p_cost
tbl_gain["profit"] = tbl_gain["revenue"] - tbl_gain["cost"]

# Для сопоставления посчитаем прибыль в случае, если бы мы опрашивали
# людей случайным образом, без использования модели
tbl_gain["profit_random"] = np.sum(tbl_gain["revenue"]) / len(tbl_gain)

# Для удобства можно округлить некоторые значения
pd.set_option('display.max_columns', None)
print(np.round(tbl_gain, 2))

# Графическая репрезентация кумулятивного выигрыша
kds.metrics.plot_cumulative_gain(df_gain["target"], df_gain["prob"])
plt.show()
plt.close()
# model  -  выигрыш модели
# random - выигрыш при случайном выборе
# wizard - выигрыши лучшей из возможных моделей

# Графическая репрезентация кумулятивного лифта
kds.metrics.plot_lift(df_gain["target"], df_gain["prob"])
plt.show()
plt.close()









