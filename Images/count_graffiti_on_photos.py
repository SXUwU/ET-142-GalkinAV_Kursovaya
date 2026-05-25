from collections import Counter
import matplotlib.pyplot as plt

file = open("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\all.csv").readlines()
file = file[1:len(file)]

for i in range(len(file)):
    s = file[i].index(",")
    
    file[i] = file[i][:s]
    
count_photos = dict(Counter(file))

num_of_photos = len(count_photos)

frequency_distribution = dict(Counter(count_photos.values()))

x_values = sorted(frequency_distribution.keys())
y_values = [frequency_distribution[x] for x in x_values]

plt.figure(figsize=(10, 6))

plt.bar(x_values, y_values, color='coral', edgecolor='black', zorder=2)

plt.xlabel('Количество граффити на фото')
plt.ylabel('Количество фотографий')
plt.title('Распределение количества объектов на фотографиях')

plt.xticks(x_values)

plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

plt.tight_layout()
plt.show()