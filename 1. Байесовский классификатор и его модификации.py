# ---------------------------------------------------------
# Машинное обучение для экономистов
# Тема - Байесовский классификатор и его обобщения
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
from sklearn.naive_bayes import CategoricalNB             # наивный байесовский классификатор
from sklearn.model_selection import train_test_split      # разделение выборки на 
                                                          # обучающую и тестовую
from sklearn.model_selection import cross_val_score       # кросс-валидация
from sklearn.model_selection import KFold                 # разбиение на фолды
from sklearn.utils import shuffle                         # случайная перестановка
import bnlearn                                            # байесовские сети
import openpyxl                                           # работа с excel
import d3blocks                                           # интерактивные графики
import random                                             # случайные числа            

# Рассматриваемые методы
# 1. Байесовский классификатор (b - bayes)
# 2. Наивный Байесовский классификатор I(nb - naive bayes)
# 3. Байесовские сети I(bn - bayesian network)

# Рассматриваемые техники сравнения качества моделей
# 1. Внутривыборочная точность прогноза
# 2. Вневыборочная точность прогноза
# 3. Кросс-валидация

# ---------------------------------------------------------
# Загрузка и первичный анализ данных
# ---------------------------------------------------------

# Загрузим данные
df = pd.read_excel("E:\Преподавание\Машинное обучение\Данные\Дефолты.xlsx")

# Посмотрим первые несколько строк в данных
df.head(5)

# Для простоты построим модель предсказания дефолта, рассматривая
# в качестве признаков лишь семейный и трудовой статусы
p_married = np.mean(df["married"])   # P(Брак = 1)
p_work = np.mean(df["work"])         # P(Работа = 1)
print([p_married, p_work])

# Изучим совместное распределение брака и работы
counts_married_work = pd.crosstab(df["married"], df["work"])

# Сохраним число наблюдений
n = df.index.size

# Посмотрим на совместное распределение
print(counts_married_work)                 # количество
print(counts_married_work / n)             # доли (оценки совместных вероятностей)

# Совместное распределение, условное на дефолт
pd.crosstab([df["married"], df["work"]], df["default"])

# ---------------------------------------------------------
# Байесовский классификатор
# Ручная реализация на простом примере
# ---------------------------------------------------------

# Оценим вероятности с помощью Байесовского классификатора
  # P(Дефолт = 1 | Брак = 1, Работа = 1)
p11 = np.mean(df.loc[(df["married"] == 1) & (df["work"] == 1), "default"])
  # P(Дефолт = 1 | Брак = 1, Работа = 0)
p10 = np.mean(df.loc[(df["married"] == 1) & (df["work"] == 0), "default"])
  # P(Дефолт = 1 | Брак = 0, Работа = 1)
p01 = np.mean(df.loc[(df["married"] == 0) & (df["work"] == 1), "default"])
  # # P(Дефолт = 1 | Брак = 0, Работа = 0)
p00 = np.mean(df.loc[(df["married"] == 0) & (df["work"] == 0), "default"])

# Предсказываем дефолт тем, у кого условная вероятность не меньше 0.5
print([p11, p10, p01, p00])

# Разделим целевую переменную и признаки
target = df.loc[:, ['default']]            # целевая переменная
features = df.loc[:, ["married", "work"]]  # матрица признаков
target = np.squeeze(target)                # преобразуем из вектора столбца 
                                           # в одномерный массив

# Оценим точность Байесовского классификатора

# Назначим каждому наблюдения соответствующую ему 
# вероятность P(Дефолт = 1 | Брак, Работа)
prob_b = np.zeros(n)                                   
prob_b[(df["married"] == 1) & (df["work"] == 1)] = p11
prob_b[(df["married"] == 1) & (df["work"] == 0)] = p10
prob_b[(df["married"] == 0) & (df["work"] == 1)] = p01
prob_b[(df["married"] == 0) & (df["work"] == 0)] = p00
print(prob_b[1:10])                                    # вероятности дефолта для 
                                                       # первых десяти индивидов

# Предскажем Дефолт тем, у кого вероятность не меньше 0.5
prediction_b = np.array(prob_b >= 0.5, dtype = int)

print(prediction_b[1:10])                              # прогнозы дефолта для 
                                                       # первых десяти индивидов

# Оценим точность внутривыборочного прогноза 
# Байесовского классификатора
ACC_b = np.mean(target == prediction_b)
print(ACC_b)

# ---------------------------------------------------------
# Наивный Байесовский классификатор
# Ручная реализация расчета одной из вероятностей
# ---------------------------------------------------------

# Воспользуемся наивным Байесовским классификатором,
# чтобы оценить P(Дефолт = 1 | Брак = 1, Работа = 0)

# Оценим априорную вероятность P(Дефолт = 1)
p_default = np.mean(df["default"])

# Оценим факторы
  # P(Брак = 1   | Дефолт = 1)
p_married1_default1 = np.mean(df.loc[df["default"] == 1, "married"])                      
  # P(Работа = 1 | Дефолт = 1)   
p_work1_default1 = np.mean(df.loc[df["default"] == 1, "work"])                            
  # P(Брак = 1   | Дефолт = 0)
p_married1_default0 = np.mean(df.loc[df["default"] == 0, "married"])                      
  # P(Работа = 1 | Дефолт = 0)   
p_work1_default0 = np.mean(df.loc[df["default"] == 0, "work"])                            
  # P(Брак = 0   | Дефолт = 1)   = 1 - P(Брак = 1 | Дефолт = 1)
p_married0_default1 = 1 - p_married1_default1                                             
p_work0_default1 = 1 - p_work1_default1
  # P(Брак = 0   | Дефолт = 0)   = 1 - P(Брак = 1 | Дефолт = 0)
p_married0_default0 = 1 - p_married1_default0                                 
  # P(Работа = 0 | Дефолт = 0)   = 1 - P(Работа = 1 | Дефолт = 0)   
p_work0_default0 = 1 - p_work1_default0                                                   

# Числитель P(Дефолт = 1 | Брак = 1, Работа = 0) = 
# = P(Брак = 1 | Дефолт = 1) *  P(Работа = 0 | Дефолт = 1) * P(Дефолт = 1)
p_default1_married1_work0_num = (p_married1_default1 * 
                                 p_work0_default1 * 
                                 p_default)      

# Числитель P(Дефолт = 0 | Брак = 1, Работа = 0) =
# = P(Брак = 1 | Дефолт = 0) *  P(Работа = 0 | Дефолт = 0) * P(Дефолт = 0)
p_default0_married1_work0_num = (p_married1_default0 * 
                                 p_work0_default0 * 
                                 (1 - p_default))

# Знаменатель вероятности P(Дефолт = 0 | Брак = 1, Работа = 0)
# P(Брак = 1, Работа = 0)
p_married1_work0 = (p_default1_married1_work0_num + 
                    p_default0_married1_work0_num)

# Условная вероятность дефолта
# P(Дефолт = 1 | Брак = 1, Работа = 0)
p_default1_married1_work0 = p_default1_married1_work0_num / p_married1_work0

# Условная вероятность отсутствия дефолта
# P(Дефолт = 0 | Брак = 1, Работа = 0)
p_default0_married1_work0 = p_default0_married1_work0_num / p_married1_work0

# Если условная вероятность дефолта не меньше 0.5, то прогнозируем дефолт для
# безработных людей в браке
print(p_default1_married1_work0)

# ---------------------------------------------------------
# Наивный Байесовский классификатор
# Автоматическая реализация
# ---------------------------------------------------------

# Обучим наивный Байесовский классификатор 
# с помощью встроенной функции
nb = CategoricalNB(force_alpha = True,     # подготавливаем модель
                   alpha = 0)              # alpha - параметр сглаживания Лапласа
nb.fit(features, target)                   # обучаем модель

# Оцениваем вероятности
prob_nb = nb.predict_proba(features)
prob_nb[0:10:, 0]                          # оценки P(Y = 0 | X = x)
prob_nb[0:10:, 0]                          # оценки P(Y = 1 | X = x)

# Предсказания
prediction_nb = nb.predict(features)       # I(P(Y = 1 | X = x) > 0.5)

# Оценим точность наивного Байесовского 
# класификатора внутривыборочно
ACC_nb = np.mean(target == prediction_b) # вручную как долю случаев, когда 
                                         # значение целевой переменной совпало 
                                         # с прогнозом
ACC_nb = nb.score(features, target)      # автоматически

# Сравним точность Байесовского классификатора и 
# Наивного Байесовского классификатора
print(ACC_b, ACC_nb)

# Оценим вероятность для конкретного наблюдения
observation = pd.DataFrame(data = {'married': [1], # Брак = 1, Работа = 0
                                   'work': [0]}) 
p_observation = nb.predict_proba(observation)
p_observation[:, 0] # P(Дефолт = 1 | Брак = 1, Работа = 0)
p_observation[:, 1] # P(Дефолт = 0 | Брак = 1, Работа = 0)

# Сравним результаты ручных  расчетов с автоматической функцией
print([p_default0_married1_work0, p_observation[0, 0]])

# ---------------------------------------------------------
# Сопоставление результатов Байесовского классификатора и 
# наивного Байесовского классификатора
# ---------------------------------------------------------

# Создадим таблицу с результатами,
# содержащую данные, вероятности 
# и прогнозы
table = features.copy()                 # признаки
table["default"] = target               # целевая переменная
table["prediction_b"] = prediction_b    # прогнозы Байесовского классификатора
table["prob_b"] = prob_b                # вероятности Байесовского 
                                        # классификатора
table["prediction_nb"] = prediction_nb  # прогнозы наивного Байесовского
                                        # классификатора
table["prob_nb"] = prob_nb[:, 1]        # вероятности наивного Байесовского 
                                        # классификатора
table.head(10)

# ---------------------------------------------------------
# Обучение наивного Байесовского классификатора с 
# дополнительным признаком на уровень образования
# ---------------------------------------------------------

# Изучим возможность добавления признака образования
pd.crosstab([df["married"], df["work"]], df["educ"])

# Проблема - некоторые комбинации признаков не наблюдаются, что 
# не позволяет применить Байесовский классификатор, по крайней
# мере без сглаживания
# Решение - воспользоваться наивным Байесовским классификатором

# Повторим оценивание наивного Байесовского
# классификатора добавив признак на 
# уровень образования
features2 = features.copy()                 # копируем признаки исходной модели
features2["educ"] = df["educ"]              # и добавляем к ним признак образования
nb2 = CategoricalNB(force_alpha = True,     # подготавливаем модель
                    alpha = 1)              # alpha - параметр сглаживания Лапласа
nb2.fit(features2, target)                  # и обучаем ее на новых признаках
prediction2 = nb2.predict(features2)        # предсказываем знаение дефолта
ACC_nb2 = nb2.score(features2, target)      # оцениваем точность 
                                            # внутривыборочных прогнозов
print([ACC_nb, ACC_nb2])                    # сравниваем точность прогнозов 
                                            # исходной и новой моделей с 
                                            # наивным прогнозом
                                            
# ---------------------------------------------------------
# Анализ точности наивного Байесовского классификатора с 
# применением тестовой выборки
# ---------------------------------------------------------

# Разделим выборку на обучающую и тестовую
# с помощью автоматической функции
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.2, random_state = 777)
# test size    - доля тестовой выборки, например, если test_size = 0.2 и у вас 
#                100 наблюдений, то в тестовую выборку войдет 20 наблюдений, 
#                а в обучающую попадет 80 наблюдений.
# random_state - параметр, изменяя который можно случайным образом изменять 
#                наблюдения, попадающие в тестовую и обучающую выборки.

# Убедимся, что обучающая и тестовая выборки имеют верные пропорции
print(features_train.index.size, features_test.index.size) # признаки
print(target_train.index.size, target_test.index.size)     # целевая переменная

# Оценим первую модель (без образования) на обучающей выборке
nb_train = CategoricalNB(force_alpha = True, alpha = 1)  
nb_train.fit(features_train, target_train)

# Оценим точность прогноза на тестовой выбокре
ACC_nb_test = nb_train.score(features_test, target_test)

# Сравним точность прогноза на обучающей и тестовой выборках
ACC_nb_train = nb_train.score(features_train, target_train)
print([ACC_nb_train, ACC_nb_test])

# ---------------------------------------------------------
# Кросс-валидация наивных Байесовских классификаторов
# ---------------------------------------------------------

# Кросс-валидация

# Первая модель (без образования)
ACC_CV_nb = cross_val_score(nb,                 # модель
                            features, target,   # признаки и целевая переменная
                            cv = 5)             # количество фолдов (folds)
ACC_CV_total_nb = np.mean(ACC_CV_nb)            # средняя точность по фолдам 
                                                                       
print(ACC_CV_nb)                                # вектор точностей посчитанны
                                                # для каждого фолда  
# Вторая модель (с образованием)
ACC_CV_nb2 = cross_val_score(nb2,               # модель
                             features2, target, # признаки и целевая переменная
                             cv = 5)            # количество фолдов (folds)
ACC_CV_total_nb2 = np.mean(ACC_CV_nb2)          # средняя точность по фолдам 
  
print(ACC_CV_nb2)                               # вектор точностей посчитанны
                                                # для каждого фолда      

# Сопоставим результаты
print([ACC_CV_total_nb, ACC_CV_total_nb2])

# ---------------------------------------------------------
# Использование биннинга в отношении непрерывной переменной
# ---------------------------------------------------------

# Для добавления переменной на доход разобьем ее по квантилям
df['income_bin'] = pd.qcut(df['income'],   # разбиваемая переменная
                           q = 3)          # число квантилей (равных разбиений)       
df['income_bin'].value_counts()            # распределение значений

# Преобразуем переменную в числовую за счет
# добавления переменной labels
df['income_bin'] = pd.qcut(df['income'], q = 3, labels = range(0, 3))

# Включим соответствующую переменную в анализ
# и проведем кросс-валидацию
features3 = features2.copy()                    # копируем признаки второй модели
features3["income_bin"] = df["income_bin"]      # и добавляем к ним признак дохода
nb3 = CategoricalNB(force_alpha = True,         # подготавливаем модель
                    alpha = 1)                  # alpha - параметр сглаживания Лапласа
nb3.fit(features3, target)                      # и обучаем ее на новых признаках
prediction3 = nb3.predict(features3)            # предсказываем знаение дефолта
ACC_CV_nb3 = cross_val_score(nb3,               # модель
                             features3, target, # признаки и целевая переменная
                             cv = 5)            # количество фолдов (folds)
ACC_CV_total_nb3 = np.mean(ACC_CV_nb3)          # средняя точность по фолдам
print([ACC_CV_total_nb,                         # сопоставление результатов
       ACC_CV_total_nb2,
       ACC_CV_total_nb3])

# ---------------------------------------------------------
# Байесовская сеть
# ---------------------------------------------------------

# Подробно о разных способах задать DAG
# # https://www.bnlearn.com/examples/dag/

# Создадим DAG
edges = [("work", "income_bin"),     # (откуда стрелочка, куда стрелочка)
         ("educ", "income_bin"),
         ("income_bin", "default"),
         ("educ", "default"),
         ("work", "default"),
         ("married", "default")]
DAG = bnlearn.make_DAG(edges, methodtype = 'bayes')

# Оценим условные вероятности в Нашей модели
bn = bnlearn.parameter_learning.fit(DAG, df, methodtype='bayes')

# Визуализируем DAG
bnlearn.plot(bn, interactive = True)

# Предскажем вероятности дефолта
predict_bn = bnlearn.predict(bn, df = features3, variables = ["default"])

# Посмотрим на матрицу.
predict_bn.head(10)
# Важно: столбец 'p' отражает не вероятности 1, а вероятности
# наиболее вероятных значений

# Достанем предсказания Дефолтов
prediction_bn = predict_bn.loc[:, "default"]

# Достанем оценки условных вероятностей дефолта
# с учетом того, что изначально мы имеем лишь вероятности
# наиболее вероятных категорий
prob_nb = predict_bn.loc[:, "p"]
prob_nb[prediction_bn == 0] = 1 - prob_nb[prediction_bn == 0]

# Оценим внутривыборочную точность
ACC_bn = np.mean(target == prediction_bn)
print([ACC_bn])

# ---------------------------------------------------------
# Проверка точности Байесовской сети на тестовой выборке
# и сравнение с навиным Байесовским классификатором
# ---------------------------------------------------------

# Разобьем данные на тренеровочные и тестовые
features3_train, features3_test, target_train, target_test = train_test_split(
    features3, target, test_size = 0.2, random_state = 777)
    
# Поскольку для обучения Байесовской сети нужно подавать сразу все
# данные, без разбиения на признаки и целевую переменную, агрегируем
# их в единный датафрейм
df_train = features3_train.copy()
df_train["default"] = target_train

# Оценим параметры модели на тренеровочной выборке
bn_train = bnlearn.parameter_learning.fit(DAG, df_train, methodtype = 'bayes')

# Получим прогнозы по тестовой выборке
predict_bn_test = bnlearn.predict(bn, df = features3_test, 
                                  variables = ["default"])
prediction_bn_test = np.array(predict_bn_test.loc[:, "default"])

# Оценим точность прогноза по тестовой выборке и сравним
# с внутривыборочным прогнозом
ACC_bn_test = np.mean(target_test == prediction_bn_test)
print([ACC_bn_test, ACC_bn])

# Сравним точность вневыборочно прогноза с Байесовской сетью
nb3.fit(features3_train, target_train)
prediction3_test = nb3.predict(features3_test)
ACC_nb3_test = nb3.score(features3_test, target_test)
print([ACC_bn_test, ACC_nb3_test])

# ---------------------------------------------------------
# Кросс-валидация Байесовской сети
# ---------------------------------------------------------

# Часто в различных пакетах не реализована автоматическая
# процедура кросс-валидации, поэтому ее необходимо
# запрограммировать самостоятельно

# Для воспроизводимости
random.seed(123)

# Реализуем 5-fold кросс-валидацию
n_folds = 5

# Рандомизируем порядок наблюдений
ind_random = np.array(shuffle(range(0, n)))

# Вектор, в котором будут храниться результаты кросс-валидации
ACC_CV_bn = np.zeros(n_folds)

# Сохраним информацию о числе наблюдений в каждом фолде
ind_fold = np.split(ind_random, n_folds)
# Подумайте, как реализовать данный метод в случае, когда
# у вас оказывается неравеное число наблюдений в каждом фолде

# Пройдемся циклом отдельно по каждому из фолдов
for i in range(0, n_folds):
  # Тестовая выборка
  features_test_cv = features3.iloc[ind_fold[i]]
  target_test_cv = target[ind_fold[i]]
  # Обучающая выборка
  features_train_cv = features3.iloc[~df.index.isin(ind_fold[1])]
  target_train_cv = target[~df.index.isin(ind_fold[1])]
  # Объединение признаков и целевой переменной обучающей выборки
  df_train_cv = features_train_cv
  df_train_cv["default"] = target_train
  # Оцениваем параметры на обучающей выборке
  bn_train_cv = bnlearn.parameter_learning.fit(DAG, df_train, 
                                               methodtype = 'bayes')
  # Получим прогнозы по тестовой выборке
  predict_bn_test_cv = bnlearn.predict(bn_train_cv, df = features_test_cv, 
                                       variables = ["default"])
  prediction_bn_test_cv = np.array(predict_bn_test_cv.loc[:, "default"])
  # Оценим точность прогноза по тестовой выборке и сравним
  # с внутривыборочным прогнозом
  ACC_CV_bn[i] = np.mean(target_test_cv == prediction_bn_test_cv)

# Посмотрим на результаты по фолдам
print(ACC_CV_bn)

# Усредним результаты
ACC_CV_total_bn = np.mean(ACC_CV_bn)
print(ACC_CV_total_bn)

# ---------------------------------------------------------
# Обучение структуры Байесовской сети
# ---------------------------------------------------------

# Методы обучения структуры
# https://erdogant.github.io/bnlearn/pages/html/Structure%20learning.html

# Подберем оптимальную структуры графа на обучающей выборе
bn2_train_structure = bnlearn.structure_learning.fit(df_train, 
                                                     methodtype = 'hc', 
                                                     scoretype = 'bic')
# Аргументы
# methodtype - метод поиска структуры Байесовской сети, например, значение
#              'hc' соответствует hill clib search
# scoretype  - критерий качества структуры сети, используемый при поиске
#              ее оптимальной формы
                                           
                                           
# Посмотрим на результат
bnlearn.plot(bn2_train_structure, interactive = True)

# Сохраним найденный DAG
DAG2 = bnlearn.make_DAG(bn2_train_structure)

# Оценим модель с подобранным DAG
bn2_train = bnlearn.parameter_learning.fit(DAG2, df_train, 
                                           methodtype = 'bayes')

# Получим прогнозы по тестовой выборке
predict_bn2_test = bnlearn.predict(bn2_train, df = features3_test, 
                                   variables = ["default"])
prediction_bn2_test = np.array(predict_bn2_test.loc[:, "default"])

# Оценим точность прогноза по тестовой выборке и сравним
# с прогнозом модели с графом, подобранным вручную
ACC_bn2_test = np.mean(target_test == prediction_bn2_test)
print([ACC_bn_test, ACC_bn2_test])
