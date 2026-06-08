import pandas as pd

file = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\all.csv")

print(file.isnull().sum())