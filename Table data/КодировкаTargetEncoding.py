import pandas as pd
from category_encoders import TargetEncoder

# 1. Загрузка данных
df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 1\\fraud_detection_dataset.csv")

print(df[['is_fraud', 'merchant_category', 'device_type', 'location']].head())
print("|")

# 2. Определение признаков и целевой переменной
cat_cols = ['merchant_category', 'device_type', 'location']
target = 'is_fraud'

# 3. Инициализация энкодера
# Параметр smoothing помогает бороться с переобучением на редких категориях
encoder = TargetEncoder(cols=cat_cols, smoothing=10.0)

# 4. Применение кодировки
# Создаем копию, чтобы не менять исходный датафрейм
df_encoded = df.copy()

# Заменяем исходные текстовые категории на закодированные числа
df_encoded[cat_cols] = encoder.fit_transform(df_encoded[cat_cols], df_encoded[target])

# Посмотрим на результат
print(df_encoded[['is_fraud', 'merchant_category', 'device_type', 'location']].head())