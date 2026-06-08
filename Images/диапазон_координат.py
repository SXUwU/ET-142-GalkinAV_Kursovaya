import pandas as pd


df = pd.read_csv(r"C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 3\\all.csv")

print(f"xmin (максимальное и минимальное значение): {df["xmin"].max()} и {df["xmin"].min()}")
print(f"ymin (максимальное и минимальное значение): {df["ymin"].max()} и {df["ymin"].min()}")
print(f"xmax (максимальное и минимальное значение): {df["xmax"].max()} и {df["xmax"].min()}")
print(f"ymax (максимальное и минимальное значение): {df["ymax"].max()} и {df["ymax"].min()}")