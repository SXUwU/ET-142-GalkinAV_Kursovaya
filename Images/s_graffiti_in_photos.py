import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Читаем датасет
df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\all.csv")

# Задаем константы размеров фото (высота и ширина из датасета)
PHOTO_WIDTH = 720
PHOTO_HEIGHT = 960
PHOTO_AREA = PHOTO_WIDTH * PHOTO_HEIGHT

# Словари для хранения результатов по каждому фото
true_areas = {}
overlap_percents = {}

# 2. Группируем по фотографиям
for filename, group in df.groupby('filename'):
    # Создаем матрицу нулей размером с фото (Высота x Ширина)
    # uint8 достаточно, так как мы просто считаем слои боксов
    mask = np.zeros((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=np.uint8)
    
    for _, row in group.iterrows():
        # Ограничиваем координаты границами изображения (на случай выбросов в разметке)
        xmin = max(0, int(row['xmin']))
        ymin = max(0, int(row['ymin']))
        xmax = min(PHOTO_WIDTH, int(row['xmax']))
        ymax = min(PHOTO_HEIGHT, int(row['ymax']))
        
        # "Закрашиваем" область бокса: увеличиваем значение пикселей на 1
        mask[ymin:ymax, xmin:xmax] += 1
        
    # Истинная площадь граффити — это количество пикселей, которые были закрашены хотя бы 1 раз
    union_area = np.sum(mask > 0)
    true_areas[filename] = union_area
    
    # Площадь наложений — это пиксели, где лежат 2 и более бокса одновременно
    overlap_area = np.sum(mask > 1)
    
    # 3. Считаем процент перекрытия боксов для конкретного фото
    # Отношение перекрывающихся пикселей к реальной площади граффити
    if union_area > 0:
        overlap_pct = (overlap_area / union_area) * 100
    else:
        overlap_pct = 0.0
        
    overlap_percents[filename] = overlap_pct

# 4. Преобразуем результаты в структуры Pandas для удобства
true_areas_series = pd.Series(true_areas)
overlap_percents_series = pd.Series(overlap_percents)

# Вычисляем процент от общей площади фотографии
percent_per_photo = (true_areas_series / PHOTO_AREA) * 100

# 5. Выводим статистику
print(f"В среднем граффити занимают {percent_per_photo.mean():.2f}% площади на фотографиях.")
print(f"Максимально занятая площадь на одном фото: {percent_per_photo.max():.2f}%")
print(f"Максимальный процент наложения боксов друг на друга: {overlap_percents_series.max():.2f}%")
print(f"Файл с максимальным наложением: {overlap_percents_series.idxmax()}")


# --- 6. Построение графика ---
plt.figure(figsize=(12, 6))

# Строим гистограмму распределения
plt.hist(percent_per_photo, bins=5, color='mediumaquamarine', edgecolor='black')

# Добавляем линию среднего значения
average_percent = percent_per_photo.mean()
plt.axvline(average_percent, color='red', linestyle='dashed', linewidth=2, 
            label=f'Среднее: {average_percent:.2f}%')

# Настраиваем подписи
plt.title('Распределение реальной площади граффити на фотографиях (с учетом наложений)')
plt.xlabel('Процент занятой площади (%)')
plt.ylabel('Количество фотографий')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Показываем график
plt.show()