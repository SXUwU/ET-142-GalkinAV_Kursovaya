import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Вычисление количества слов
def count_clean_words(text):
    if pd.isna(text):
        return 0
    
    words = text.split()
    
    clean_words = [w for w in words if w not in ['.', '', '!', '?', ',', '$']]
    return len(clean_words)


file = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 4\\enron_spam_data.csv")

# Пропуски
count_strings = file.count()
print(file.head(5))

nl = pd.DataFrame(data=file.isnull().sum(), columns=["Количество пропусков"])

percent = []

for i in range(len(nl)):
    percent.append((nl.iloc[i]/count_strings.iloc[i])*100)

nl["Доля пропусков (%)"] = percent

print(nl)

print("-"*40)

print(nl)

# Подсчет количества сообщений по их классу
file["Spam/Ham"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")

disbalance = file["Spam/Ham"].value_counts()["spam"] - file["Spam/Ham"].value_counts()["ham"]


plt.xlabel(f"Классы. Дисбаланс составляет {disbalance}")
plt.ylabel("Количество")
plt.xticks(rotation=0)
plt.show()


file["Subject_len"] = file["Subject"].str.len()
file["Message_len"] = file["Message"].str.len()
file["Words_in_Message"] = file["Message"].apply(count_clean_words)
file["Words_in_Subject"] = file["Subject"].apply(count_clean_words)

means = file.groupby('Spam/Ham')[['Subject_len', 'Words_in_Subject', 'Message_len', 'Words_in_Message']].mean()


# График со средними значениями количества слов и длин сообщений и их тем
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


means[['Message_len', 'Words_in_Message']].plot(kind='bar', ax=ax1, color=['#4C72B0', '#55A868'])
ax1.set_title('Метрики сообщения (Message)')
ax1.set_ylabel('Среднее количество')
ax1.set_xticklabels(['Обычное', 'Спам'], rotation=0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)


means[['Subject_len', 'Words_in_Subject']].plot(kind='bar', ax=ax2, color=["#C44B4F", '#8172B2'])
ax2.set_title('Метрики заголовка (Subject)')
ax2.set_ylabel('Среднее количество')
ax2.set_xticklabels(['Обычное', 'Спам'], rotation=0)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



