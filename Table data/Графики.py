import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\fraud_detection_dataset.csv")


print("-"*20 + "Пример данных из датасета и информация" + "-"*20)
print(df.head())
print(df.info())
print("-"*100)

print("-"*20 + "Количество пропусков по каждому столбцу" + "-"*20)
print(df.isnull().sum())
print("-"*100)

print("-"*20 + "Распределение классов 'is_fraud'" + "-"*20)
print(df['is_fraud'].value_counts(normalize=True) * 100)
print("-"*100)

plt.figure()
sns.countplot(x='is_fraud', data=df)
plt.title('Распределение мошеннических и обычных транзакций')
plt.xlabel('Мошенничество (0 - Нет, 1 - Да)')
plt.ylabel('Количество транзакций')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

sns.histplot(df['user_age'], bins=30, ax=axes[0], kde=True)
axes[0].set_title('Распределение возраста пользователя')
axes[0].set_xlabel('Возраст')

sns.histplot(df['hour_of_day'], bins=24, ax=axes[1], kde=False)
axes[1].set_title('Распределение транзакций по часам дня')
axes[1].set_xlabel('Час дня')

plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(14, 10))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица числовых признаков')
plt.show()


plt.figure()
sns.boxplot(x='is_fraud', y='amount', data=df[df['amount'] < 1000]) 
plt.title('Сравнение суммы транзакции (мошенничество vs обычные)')
plt.xlabel('Мошенничество')
plt.ylabel('Сумма транзакции')
plt.yscale('log') 
plt.show()


fraud_by_hour_day = df[df['is_fraud'] == 1].groupby(['hour_of_day', 'day_of_week']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(fraud_by_hour_day, annot=True, fmt='d', cmap='Reds')
plt.title('Тепловая карта мошеннических транзакций по часу дня и дню недели')
plt.xlabel('День недели')
plt.ylabel('Час дня')
plt.show()


plt.figure(figsize=(12, 4))
fraud_by_category = df[df['is_fraud'] == 1]['merchant_category'].value_counts()
sns.barplot(x=fraud_by_category.values, y=fraud_by_category.index, palette='viridis')
plt.title('Количество мошеннических транзакций по категориям транзакций')
plt.xlabel('Количество мошенничеств')
plt.show()
