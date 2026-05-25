import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# 1. Загрузка данных
# Считываем наш CSV-файл
df = pd.read_csv("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Пункт 4\\enron_spam_data.csv")

# Удаляем строки, где текст письма (Message) пустой, чтобы избежать ошибок
df = df.dropna(subset=['Message'])

# 2. Разделение данных на легитимные письма (ham) и спам (spam)
# Приводим значения в колонке 'Spam/Ham' к нижнему регистру для надежности
ham_emails = df[df['Spam/Ham'].str.lower() == 'ham']['Message']
spam_emails = df[df['Spam/Ham'].str.lower() == 'spam']['Message']

# Объединяем все письма каждой категории в один гигантский текст
ham_text = " ".join(ham_emails)
spam_text = " ".join(spam_emails)

# 3. Настройка стоп-слов (слов-паразитов)
# Берем стандартный набор английских стоп-слов (the, and, to, i и т.д.)
stopwords = set(STOPWORDS)

# Корпоративная почта Enron содержит много специфического "шума".
# Добавим слова вроде 'subject', 'cc', 'enron', 'ect', 'hou', которые часто встречаются,
# но не несут важного смысла для понимания тематики.
custom_stopwords = ['subject', 'cc', 'enron', 'ect', 'hou', 'forwarded', 'com', 'please', 'thanks']
stopwords.update(custom_stopwords)

# 4. Генерация облака тегов для HAM (надежные письма)
# Для наглядности сделаем белый фон
wordcloud_ham = WordCloud(
    width=800, 
    height=800,
    background_color='white',
    stopwords=stopwords,
    min_font_size=10
).generate(ham_text)

# 5. Генерация облака тегов для SPAM (спам)
# Для контраста сделаем черный фон
wordcloud_spam = WordCloud(
    width=800, 
    height=800,
    background_color='black',
    stopwords=stopwords,
    min_font_size=10
).generate(spam_text)

# 6. Визуализация и вывод на экран
# Создаем общее окно для двух графиков (1 row, 2 columns)
plt.figure(figsize=(16, 8))

# Левый график — Облако для Ham
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.title('Облако тегов: Легитимные письма (Ham)', fontsize=18, pad=15)
plt.axis("off")  # Отключаем оси координат, они тут не нужны

# Правый график — Облако для Spam
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.title('Облако тегов: Спам (Spam)', fontsize=18, pad=15)
plt.axis("off")  # Отключаем оси координат

# Подгоняем графики, чтобы они не перекрывали друг друга
plt.tight_layout(pad=2)

# Показываем результат
plt.show()