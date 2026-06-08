import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def show_images_with_bboxes(csv_path, images_dir, num_images_to_show=5):
    # Чтение данных разметки
    df = pd.read_csv(csv_path)
    
    # Получаем список уникальных изображений из датафрейма
    unique_images = df['filename'].unique()
    
    # Ограничиваем количество выводимых изображений для удобства
    for idx, img_name in enumerate(unique_images[:num_images_to_show], start=1):
        img_path = os.path.join(images_dir, img_name)
        
        # Проверяем, существует ли файл в папке
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            continue
            
        # Загружаем изображение
        image = Image.open(img_path)
        
        # Создаем фигуру для отрисовки
        fig, ax = plt.subplots(1, figsize=(8, 10))
        ax.imshow(image)
        
        # Выбираем все bounding boxes для текущего изображения
        # (их может быть несколько на одной фотографии)
        boxes = df[df['filename'] == img_name]
        
        for _, row in boxes.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            # Вычисляем ширину и высоту прямоугольника
            box_width = xmax - xmin
            box_height = ymax - ymin
            
            # Создаем прямоугольник
            rect = patches.Rectangle(
                (xmin, ymin), box_width, box_height, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            
            # Добавляем прямоугольник на график
            ax.add_patch(rect)
        
        # Добавляем нумерацию в качестве заголовка вместо названий файлов
        ax.set_title(f"Изображение {idx}")
        
        # Отключаем оси для более чистого визуального представления
        ax.axis('off') 
        plt.show()

# Пути к файлам (относительно папки, из которой запускается скрипт)
train_csv = os.path.join('bounding_boxes', 'train_labels.csv')
train_images_dir = os.path.join('images', 'train')

test_csv = os.path.join('bounding_boxes', 'test_labels.csv')
test_images_dir = os.path.join('images', 'test')

# # Запуск функции для тренировочной выборки
# print("Отображение тренировочной выборки:")
# show_images_with_bboxes("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\imagesofgraffiti\\3238357\\bounding_boxes\\train_labels.csv", 
#                         "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\imagesofgraffiti\\3238357\\images\\train", num_images_to_show=20)

# Чтобы посмотреть тестовую выборку, можно раскомментировать код ниже:
print("Отображение тестовой выборки:")
show_images_with_bboxes("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\imagesofgraffiti\\3238357\\bounding_boxes\\test_labels.csv", 
                        "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\imagesofgraffiti\\3238357\\images\\test", num_images_to_show=20)