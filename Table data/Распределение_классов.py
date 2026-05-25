import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 1\\fraud_detection_dataset.csv")

is_f = file["is_fraud"].value_counts()
is_f = is_f.rename({0: "Обычные транзакции", 1: "Мошеннические транзакции"})

is_f.plot(kind = "bar", rot = 0, color=['#4C72B0', '#C44E52'])
plt.show()

print(is_f)