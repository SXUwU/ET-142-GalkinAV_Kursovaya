import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\imagesofgraffiti\\3238357\\bounding_boxes\\all.csv")
print(file.describe())

file["class"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")


plt.xlabel("Класс")
plt.ylabel("Количество")
plt.xticks(rotation=45)
plt.show()

