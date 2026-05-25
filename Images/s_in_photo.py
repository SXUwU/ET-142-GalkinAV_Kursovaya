import pandas as pd
import matplotlib.pyplot as plt

# 1. Читаем датасет
df = pd.read_csv(r"C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\all.csv")

# Общая площадь одной фотографии (в пикселях)
# 720 (ширина) * 960 (высота) = 691200 пикселей
PHOTO_AREA = 720 * 960

# 2. Считаем площадь каждого отдельного граффити в пикселях
df['graffiti_area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])

# 3. Группируем по фотографиям и суммируем площадь граффити для каждой
# Метод .groupby() соберет все строки с одинаковым 'filename' 
# и сложит значения 'graffiti_area'
area_per_photo = df.groupby('filename')['graffiti_area'].sum()

# 4. Вычисляем процент площади, занятой граффити, для каждой фотографии
percent_per_photo = (area_per_photo / PHOTO_AREA) * 100

# 5. Считаем среднее значение в процентах по всему датасету
average_percent = percent_per_photo.mean()
print(f"В среднем граффити занимают {average_percent:.2f}% площади на фотографиях.")

# --- 6. Построение графика ---

plt.figure(figsize=(12, 6))

# Строим гистограмму распределения процентов
# bins=20 означает, что мы разобьем график на 20 столбиков (интервалов)
plt.hist(percent_per_photo, bins=5, color='mediumaquamarine', edgecolor='black')

# Добавляем яркую линию, которая покажет наше среднее значение
plt.axvline(average_percent, color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {average_percent:.2f}%')

# Настраиваем подписи
plt.title('Распределение площади, занятой граффити на фотографиях')
plt.xlabel('Процент занятой площади (%)')
plt.ylabel('Количество фотографий')

# Выводим легенду (чтобы было понятно, что значит красная линия)
plt.legend()

# Добавляем сетку для наглядности
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Показываем график
plt.show()