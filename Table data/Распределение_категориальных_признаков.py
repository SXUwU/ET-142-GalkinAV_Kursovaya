import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 1\\fraud_detection_dataset.csv")

# is_f = file["is_fraud"].value_counts()
# is_f = is_f.rename({0: "Обычные транзакции", 1: "Мошеннические транзакции"})

# is_f.plot(kind = "bar", rot = 0, color=['#4C72B0', '#C44E52'])
# plt.show()

# print(is_f)

sns.set_theme(style="whitegrid")

# Список целевых категориальных признаков
categorical_features = ['merchant_category', 'device_type', 'location']

# 3. Создание общего полотна (фигуры) для графиков
plt.figure(figsize=(18, 5))
 
# 4. Построение графиков в цикле
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    
    # Строим столбчатую диаграмму (обновленный синтаксис)
    sns.countplot(
        data=df, 
        x=feature, 
        order=df[feature].value_counts().index, 
        palette="viridis",
        hue=feature,      # Добавлено для устранения предупреждения
        legend=False      # Отключаем ненужную легенду
    )
    
    # Настраиваем заголовки и подписи
    plt.title(f'Распределение: {feature}', fontsize=14, pad=10)
    plt.xlabel('')
    plt.ylabel('Количество транзакций')
    
    # Поворачиваем подписи оси X, чтобы они не слипались
    plt.xticks(rotation=45, ha='right')

# 5. Вывод на экран
plt.tight_layout()
plt.show()