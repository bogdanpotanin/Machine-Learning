# ---------------------------------------------------------
# Машинное обучение для экономистов
# Тема - Решающие деревья и случайный лес
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

# Рассматриваемые методы
# 1. Решающие деревья (dt - decision tree)
# 2. Случайный лес (rf - random forest)

# Рассматриваемые техники сравнения качества моделей
# 1. Сравнение с учетом ценности прогнозов

# ---------------------------------------------------------
# Загрузка и первичный анализ данных
# ---------------------------------------------------------

# Загрузим данные
df = pd.read_excel("E:\Преподавание\Машинное обучение\Данные\Клиенты.xlsx")

# Посмотрим первые несколько строк в данных
df.head(5)

# Сохраним число наблюдений
n = df.index.size

# Разделим целевую переменную и признаки
target = df.loc[:, ['churn']]                   # целевая переменная
features = df.loc[:, df.columns.drop('churn')]  # матрица признаков
target = np.squeeze(target)                     # преобразуем из вектора столбца 
                                                # в одномерный массив

# ---------------------------------------------------------
# Решающее дерево
# ---------------------------------------------------------

# Обучим решающее дерево
dt = tree.DecisionTreeClassifier(max_depth = 3)
dt.fit(features, target)

# Визуализируем полученный результат
tree.plot_tree(dt, feature_names = features.columns)
plt.show()
plt.close()

# Оцениваем вероятности
prob_dt = dt.predict_proba(features)
prob_dt[0:10:, 0]                          # оценки P(Y = 0 | X = x)
prob_dt[0:10:, 1]                          # оценки P(Y = 1 | X = x)

# Предсказания
prediction_dt = dt.predict(features)       # I(P(Y = 1 | X = x) > 0.5)

# Оценим точность
ACC_dt = dt.score(features, target)

# ---------------------------------------------------------
# Проблема переобучения решающих деревьев
# ---------------------------------------------------------

# Сделаем максимальную глубину дерева достаточно большой
dt2 = tree.DecisionTreeClassifier(max_depth = 12)
dt2.fit(features, target)

# Оценим точность
ACC_dt2 = dt2.score(features, target)

# Сравним точность внутривыборочного прогноза с деревом, 
# обладающим меньшей максимальной глубиной
print([ACC_dt, ACC_dt2])

# Сопоставим точность вневыборочно прогноза с помощью кросс-валидации
ACC_CV_dt = cross_val_score(dt, features, target, cv = 5)   # модель dt             
ACC_CV_dt2 = cross_val_score(dt2, features, target, cv = 5) # модель dt 2        
print([np.mean(ACC_CV_dt), np.mean(ACC_CV_dt2)])

# ---------------------------------------------------------
# Случайный лес
# ---------------------------------------------------------

# Документация
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Обучим случайный лес
rf = RandomForestClassifier(max_depth = 12,         # максимальная глубина деревьев
                            max_features = "sqrt",  # число случайно выбираемых на
                                                    # каждой итерации признаков
                            max_samples = 500)      # число бутстрап итераций
rf.fit(features, target)

# Оцениваем вероятности
prob_rf = rf.predict_proba(features)
prob_rf[0:10:, 0]                          # оценки P(Y = 0 | X = x)
prob_rf[0:10:, 1]                          # оценки P(Y = 1 | X = x)

# Предсказания
prediction_rf = dt.predict(features)       # I(P(Y = 1 | X = x) > 0.5)

# Оценим точность внутри выборки
ACC_rf = rf.score(features, target)
print([ACC_dt, ACC_dt2, ACC_rf])

# Сопоставим точность вневыборочно прогноза с помощью кросс-валидации
ACC_CV_rf = cross_val_score(rf, features, target, cv = 5)       
print([np.mean(ACC_CV_dt), np.mean(ACC_CV_dt2), np.mean(ACC_CV_rf)])

# ---------------------------------------------------------
# Ошибка неотобранных элементов
# ---------------------------------------------------------

# Ошибка неотобранныэ лементов (OOB error)
rf.oob_score = True                         # укажем необходимость подсчета OOB
rf.fit(features, target)                    # обучим модель рассчитав OOB
oob_rf = rf.oob_score_                      # значение OOB

# Повторим оценивание с другим 
# числом регрессоров
rf2 = rf
rf2.max_features = 2                        # число случайно выбираемых на
                                            # каждой итерации признаков
rf2.fit(features, target)                   # обучаем новую модель
oob_rf2 = rf2.oob_score_                    # значение OOB
print([oob_rf, oob_rf2])                    # сравнваем OOB моделей

# ---------------------------------------------------------
# Оценивание выигрыша от прогноза
# ---------------------------------------------------------

# Рассмотрим прогнозы различного вида
  # автоматически
TN, FP, FN, TP = confusion_matrix(target, prediction_dt).ravel()
  # вручную
TP = np.sum((target == 1) & (prediction_dt == 1))
TN = np.sum((target == 0) & (prediction_dt == 0))
FP = np.sum((target == 0) & (prediction_dt == 1))
FN = np.sum((target == 1) & (prediction_dt == 0))
  # сохраним результат
predictions = pd.Series([TP, TN, FP, FN], index = ["TP", "TN", "FP", "FN"])
print(predictions)

# Предположим следующие цены прогнозов
# Предположим, что клиент, который не уходит, приносит 20 денежных единиц,
# и что стоимость удержания клиента составляет 1 денежные единицы
prices = pd.Series([19, 20, -1, 0], index = ["TP", "TN", "FP", "FN"])

# Оценим внутривыборочный выигрыш от прогнозов
profit = np.sum(prices * predictions)

# ---------------------------------------------------------
# Подбор оптимального порога
# ---------------------------------------------------------

# Рассмотрим модель случайного леса и для простоты
# будем работать только с обучающей выборкой

# Возможные значения порогов
thresholds_rf = np.unique(np.sort(prob_rf[:, 1]))
print(thresholds_rf)

# Число порогов
n_threshold_rf = thresholds_rf.size

# Вектор для сохранения прибылей при различных порогах
profits_rf = np.zeros(n_threshold_rf) 

# Векторы различных видо прогнозов
TP_vec = np.zeros(n_threshold_rf)
TN_vec = np.zeros(n_threshold_rf)
FP_vec = np.zeros(n_threshold_rf)
FN_vec = np.zeros(n_threshold_rf)

# Рассчитаем прибыли для различных порогов
for i in range(0, n_threshold_rf):
  # получаем предсказания при соответствующем пороге
  prediction_rf_i = (prob_rf[:, 1] >= thresholds_rf[i]).astype(int)
  # считаем количество прогнозов различного вида
  TN_vec[i], FP_vec[i], FN_vec[i], TP_vec[i] = confusion_matrix(
    target, prediction_rf_i).ravel()
  # аггрегируем результаты прогнозов
  predictions_rf_i = pd.Series([TP_vec[i], TN_vec[i], FP_vec[i], FN_vec[i]], 
                               index = ["TP", "TN", "FP", "FN"])
  # считаем прибыль
  profits_rf[i] = np.sum(prices * predictions_rf_i)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ------------------------
# Задача для программистов
# ------------------------
# Придумайте более эффективный алгоритм расчета profits, не требующий при 
# расчете TP, TN, FP и FN с новым прогнозом заново строить все прогнозы
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  
# Достанем оптимальный порог
threshold_rf = thresholds_rf[np.argmax(profits_rf)]
print(threshold_rf)
# Содержательный смысл низкого порога - обычно лучше перестраховаться
# и попытаться удержать клиента, даже если он вряд ли уйдет, чем
# столкнуться с риском его потерять, поскольку издержки на удержание
# несопоставимо меньше, чем потери от ухода клиента.

# График зависимости порога и прибыли
plt.plot(thresholds_rf, profits_rf)
plt.show()
plt.close()

# График зависимости порогов и различных типов прогнозов
  # TP
plt.plot(thresholds_rf, TP_vec)
plt.show()
plt.close()
  # TN
plt.plot(thresholds_rf, TN_vec)
plt.show()
plt.close()
  # FP
plt.plot(thresholds_rf, FP_vec)
plt.show()
plt.close()
  # FN
plt.plot(thresholds_rf, FN_vec)
plt.show()
plt.close()

# ---------------------------------------------------------
# Подбор оптимального порога кросс-валидацией
# ---------------------------------------------------------

# Для удобства создадим функцию, 
# подсчитывающую прибыли при различных порогах
def profits_thresholds(model, features, target,
                       thresholds, prices):
  prob = model.predict_proba(features)[:, 1]        # вероятности
  n_threshold = thresholds.size                     # число порогов
  profits = np.zeros(n_threshold)                   # прибыли при 
                                                    # различных порогах
  # Для каждого возможного порога
  # рассчитываем прибыль
  for i in range(0, n_threshold):
    # получаем предсказания при соответствующем пороге
    prediction_i = (prob >= thresholds[i]).astype(int)
    # считаем количество прогнозов различного вида
    TN, FP, FN, TP = confusion_matrix(target, prediction_i).ravel()
    # аггрегируем результаты прогнозов
    predictions_i = pd.Series([TP, TN, FP, FN], 
                              index = ["TP", "TN", "FP", "FN"])
    # считаем прибыль
    profits[i] = np.sum(prices * predictions_i)
  # Возвращаем результат
  return(profits)
  
# Предскажем прибыли случайного леса
profits_rf = profits_thresholds(rf, features, target,
                                thresholds_rf, prices)

# Создадим функцию для кросс-валидации
def profits_thresholds_CV(model, features, target, 
                          prices, thresholds, n_folds):
  # число порогов
  n_threshold = thresholds.size
  # суммарная прибыль по кросс-валидациям для каждого порога
  profits = np.zeros(n_threshold)
  for train_idx, test_idx in KFold(n_splits = n_folds).split(features):
     # признаки обучающей выборки
     features_train = features.loc[train_idx]
     # целевая переменная обучающей выборки
     target_train = target.loc[train_idx]
     # обучение модели
     model.fit(features_train, target_train)
     # признаки тестовой выборки
     features_test = features.loc[test_idx]
     # целевая переменная тестовой выборки
     target_test = target.loc[test_idx]
     # подсчет прибылей на тестовой выборке
     profits += profits_thresholds(model, features, target, thresholds, prices)
  return(profits / n_folds)
  
# Предскажем прибыли случайного леса
profits_CV_rf = profits_thresholds_CV(rf, features, target, 
                                      prices, thresholds, 5)

# Выберем лучший порог в соответствии с кросс-валидацией
threshold_CV_rf = thresholds_rf[np.argmax(profits_CV_rf)]

# Сравним порог обучающей выборки и кросс-валидации
print([threshold_rf, threshold_CV_rf])

# Сравним с порогом обычного решающего дерева
thresholds_dt = np.unique(np.sort(prob_dt[:, 1]))
profits_CV_dt = profits_thresholds_CV(dt, features, target, 
                                      prices, thresholds_dt, 5)
threshold_CV_dt = thresholds_dt[np.argmax(profits_CV_dt)]
print([threshold_CV_rf, threshold_CV_dt])     

# Сравним кросс-валидационную прибыль при оптимальных порогах двух методов
print([np.max(profits_CV_rf), np.max(profits_CV_dt)])   
# Примечание - подбирать оптимальный порог лучше на валидационной выборке,
#              с которой мы познакомимся позже, а на тестовой выборке сравнивать
#              прибыли при оптимальных порогах

# ---------------------------------------------------------
# Расчет важности регрессоров
# ---------------------------------------------------------

# Рассмотрим важность регрессоров на 
# основании снижения неопределенности
importances = pd.Series(rf.feature_importances_, index = features.columns)
importances.plot.bar()
plt.show()
plt.close()

# Рассмотрим важность регрессоров на 
# основании перестановок
importances2 = permutation_importance(estimator = rf, 
                                      X = features, 
                                      y = target,
                                      scoring = "accuracy")
importances2 = importances2["importances_mean"]



