import pandas as pd

file = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 4\\enron_spam_data.csv")

d=file["Message"].str.len().idxmax()

avg_len = file["Message"].str.len().mean()
print(avg_len)

print(len(file.loc[d]["Message"]))