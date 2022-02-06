#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Задание №1
data = pd.read_csv('C:/Users/Admin/Downloads/HR.csv', sep = ',')

#Задание №2
#средние значения
print(round(data['satisfaction_level'].mean(),2))
print(round(data['last_evaluation'].mean(),2))
print(round(data['number_project'].mean(),2))
print(round(data['average_montly_hours'].mean(),2))
print(round(data['time_spend_company'].mean(),2))
print(round(data['Work_accident'].mean(),2))
print(round(data['left'].mean(),2))
print(round(data['promotion_last_5years'].mean(),2))


#минимальные значения
print(round(min(data['satisfaction_level']),2))
print(round(min(data['last_evaluation']),2))
print(round(min(data['number_project']),2))
print(round(min(data['average_montly_hours']),2))
print(round(min(data['time_spend_company']),2))
print(round(min(data['Work_accident']),2))
print(round(min(data['left']),2))
print(round(min(data['promotion_last_5years']),2))

#максимальные значения
print(round(max(data['satisfaction_level']),2))
print(round(max(data['last_evaluation']),2))
print(round(max(data['number_project']),2))
print(round(max(data['average_montly_hours']),2))
print(round(max(data['time_spend_company']),2))
print(round(max(data['Work_accident']),2))
print(round(max(data['left']),2))
print(round(max(data['promotion_last_5years']),2))


#медиана
print(data['satisfaction_level'].median())
print(data['last_evaluation'].median())
print(data['number_project'].median())
print(data['average_montly_hours'].median())
print(data['time_spend_company'].median())
print(data['Work_accident'].median())
print(data['left'].median())
print(data['promotion_last_5years'].median())


#мода
print(data['satisfaction_level'].mode())
print(data['last_evaluation'].mode())
print(data['number_project'].mode())
print(data['average_montly_hours'].mode())
print(data['time_spend_company'].mode())
print(data['Work_accident'].mode())
print(data['left'].mode())
print(data['promotion_last_5years'].mode())


#дисперсия
print(round(statistics.stdev(data['satisfaction_level']),2))
print(round(statistics.stdev(data['last_evaluation']),2))
print(round(statistics.stdev(data['number_project']),2))
print(round(statistics.stdev(data['average_montly_hours']),2))
print(round(statistics.stdev(data['time_spend_company']),2))
print(round(statistics.stdev(data['Work_accident']),2))
print(round(statistics.stdev(data['left']),2))
print(round(statistics.stdev(data['promotion_last_5years']),2))

#Задание 3
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt

dt = pd.DataFrame()
data = pd.read_csv('C:/Users/Admin/Downloads/HR.csv', sep = ',')

num_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
corr_matrix = data.loc[:, num_cols].corr()
print(corr_matrix)

#Ответ: 2 наиболее скоррелированные переменные: average_montly_hours и number_project                  
# 2 наименее скоррелированные переменные: satisfaction_level и average_montly_hours  

#Задание №4
#Количество сотрудников, работающих в каждом департаменте
data[data.left !=1].groupby('department').count()['satisfaction_level']

#Задание №5
#Распределение сотрудников по зарплатам
data['salary'] = np.where((data.salary =='low'), 1, data.salary)
data['salary'] = np.where((data.salary =='medium'), 2, data.salary)
data['salary'] = np.where((data.salary =='high'), 3, data.salary)
data['salary'].plot(kind = 'hist', title = 'Распределение по зарплате', bins = 5)

#Задание №6
#Распределение сотрудников по зарплатам в каждом департаменте по отдельности
data.plot(xlabel="department", ylabel="salary", kind="bar", rot=5, fontsize=4)


#Задание №7. Проверить гипотезу, что сотрудники с высоким окладом проводят на работе больше времени, чем сотрудники с низким окладом
salary_high_time = round(data[data.salary == 'high'].mean()['average_montly_hours'],2)
salary_low_time = round(data[data.salary == 'low'].mean()['average_montly_hours'],2)

print(salary_high_time)
print(salary_low_time)
#Ответ: гипотеза неверна. 

#Задание №8.
#Показатели среди уволившихся сотрудников
#Доля сотрудников с повышением за последние 5 лет
print('Показатели среди уволившихся сотрудников:')
dolya_sotr = (data[data.left == 1].sum()['promotion_last_5years'] * 100) / data[data.left == 1].count()['satisfaction_level']
print('Доля сотрудников с повышением за последние 5 лет:', round(dolya_sotr), '%')

#Средняя степень удовлетворенности
satis_level = round(data[data.left == 1].mean()['satisfaction_level'],2)
print('Средняя степень удовлетворенности:', satis_level)


#Среднее количество проектов
av_proj = round(data[data.left == 1].mean()['number_project'])
print('Среднее количество проектов:', av_proj)

#Показатели среди работающих сотрудников
#Доля сотрудников с повышением за последние 5 лет
print('_____________________________________________')
print('Показатели среди работающих сотрудников:')
dolya_sotr_w = (data[data.left != 1].sum()['promotion_last_5years'] * 100) / data[data.left != 1].count()['satisfaction_level']
print('Доля сотрудников с повышением за последние 5 лет:', round(dolya_sotr_w), '%')

#Средняя степень удовлетворенности
satis_level_w = round(data[data.left != 1].mean()['satisfaction_level'],2)
print('Средняя степень удовлетворенности:', satis_level_w)

#Среднее количество проектов
av_proj_w = round(data[data.left != 1].mean()['number_project'])
print('Среднее количество проектов:', av_proj_w)


#Задание 9


X = data[['number_project', 'time_spend_company', 'average_montly_hours', 'Work_accident', 'promotion_last_5years', 'satisfaction_level', 'last_evaluation']]
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

lda = LinearDiscriminantAnalysis()
lda.fit(X_test, y_test).transform(X_test)
y_pred = lda.predict(X_test)

#plt.legend(loc='best', shadow=False, scatterpoints=1)


plt.scatter(X_test['number_project'], y_pred, s=80, c="r", alpha=0.1, marker="D")
plt.scatter(X_test['time_spend_company'], y_pred, s=80, c="g", alpha=0.1, marker="D")
plt.scatter(X_test['average_montly_hours'], y_pred, s=80, c="m", alpha=0.1, marker="D")
plt.scatter(X_test['Work_accident'], y_pred, s=80, c="b", alpha=0.1, marker="D")
plt.scatter(X_test['promotion_last_5years'], y_pred, s=80, c="w", alpha=0.1, marker="D", linewidths=2, edgecolors="b")
plt.scatter(X_test['satisfaction_level'], y_pred, s=80, c="r", alpha=0.1, marker="D", linewidths=2, edgecolors="b")
plt.scatter(X_test['last_evaluation'], y_pred, s=80, c="y", alpha=0.1, marker="D", linewidths=2, edgecolors="b")
plt.show()

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Точность: ' + str(accuracy_score(y_test, y_pred)))


#Ответ: модель показывает, что сотрудник уволился не на основе имеющихся факторов

