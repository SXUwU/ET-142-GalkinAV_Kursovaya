import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 1\\fraud_detection_dataset.csv")

print(df[["device_type","user_age","account_age_days","avg_transaction_amount","is_foreign_transaction","is_fraud","hour_of_day"]].head(3))

print("Максимум и минимум в столбце user_age:")
print(df["user_age"].max())
print(df["user_age"].min())

print("\nМаксимум и минимум в столбце account_age_days:")
print(df["account_age_days"].max())
print(df["account_age_days"].min())

print("\nМаксимум и минимум в столбце transaction_count_24h:")
print(df["transaction_count_24h"].max())
print(df["transaction_count_24h"].min())

print("\nМаксимум и минимум в столбце avg_transaction_amount:")
print(df["avg_transaction_amount"].max())
print(df["avg_transaction_amount"].min())

print("\nМаксимум и минимум в столбце amount:")
print(df["amount"].max())
print(df["amount"].min())

df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 1\\fraud_detection_dataset.csv")

# Функция для подсчета выбросов по методу Тьюки (IQR), который использует boxplot
def calculate_outliers(data):
    # Находим 25-й и 75-й перцентили
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Считаем межквартильный размах
    iqr = q3 - q1
    
    # Определяем границы усов графика
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Фильтруем данные, оставляя только то, что за границами
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return outliers, lower_bound, upper_bound

# 2. Разделяем суммы транзакций по классам
normal_amounts = df[df['is_fraud'] == 0]['amount']
fraud_amounts = df[df['is_fraud'] == 1]['amount']

# 3. Применяем функцию к каждому классу
normal_outliers, norm_lower, norm_upper = calculate_outliers(normal_amounts)
fraud_outliers, fraud_lower, fraud_upper = calculate_outliers(fraud_amounts)

# 4. Выводим результаты для анализа
print("--- ОБЫЧНЫЕ ТРАНЗАКЦИИ (Класс 0) ---")
print(f"Нижняя граница нормы: {norm_lower:.2f}")
print(f"Верхняя граница нормы: {norm_upper:.2f}")
print(f"Количество выбросов: {len(normal_outliers)} (из {len(normal_amounts)} транзакций)")
print(f"Максимальная сумма выброса: {normal_outliers.max() if not normal_outliers.empty else 'Нет'}")

print("\n--- МОШЕННИЧЕСКИЕ ТРАНЗАКЦИИ (Класс 1) ---")
print(f"Нижняя граница нормы: {fraud_lower:.2f}")
print(f"Верхняя граница нормы: {fraud_upper:.2f}")
print(f"Количество выбросов: {len(fraud_outliers)} (из {len(fraud_amounts)} транзакций)")
print(f"Максимальная сумма выброса: {fraud_outliers.max() if not fraud_outliers.empty else 'Нет'}")