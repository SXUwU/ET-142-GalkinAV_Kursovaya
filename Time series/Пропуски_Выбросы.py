import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def analyze_all_files(folder_path):
    # 1. Подготовка
    column_names = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]
    # Ищем все файлы .csv в папке
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    # Список для хранения результатов по каждому файлу
    results = []


    # 2. Цикл по всем файлам
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path, header=None, names=column_names)
        
        # Словарь для хранения метрик текущего файла
        file_metrics = {"file": file_name}
        
        for col in column_names:
            # --- Расчет пропусков ---
            missing_pct = df[col].isnull().mean() * 100
            file_metrics[f"{col}_missing_%"] = missing_pct
            
            # --- Расчет выбросов (3 сигмы) ---
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std
            
            outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
            file_metrics[f"{col}_outliers_count"] = outliers_count

        results.append(file_metrics)

    # 3. Создание итоговой таблицы
    summary_df = pd.DataFrame(results)
    
    return summary_df

report = analyze_all_files("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure")

# print(report["Vibration_X_outliers_count"].sum())
# print(report["Vibration_Y_outliers_count"].sum())
# print(report["Temp_Bearing_outliers_count"].sum())
# print(report["Temp_Atmosphere_outliers_count"].sum())


# Предположим, 'report' — это DataFrame, полученный из функции выше
plt.figure(figsize=(15, 6))

# Рисуем график роста количества выбросов по вибрации X
plt.plot(range(len(report)), report['Vibration_X_outliers_count'], label='Vibration X Outliers', color='blue')
plt.plot(range(len(report)), report['Vibration_Y_outliers_count'], label='Vibration Y Outliers', color='red')

plt.title("Динамика количества выбросов за весь период эксперимента (129 часов)", fontsize=14)
plt.xlabel("Время (часы)")
plt.ylabel("Количество выбросов (3-sigma)")
plt.legend()
plt.grid(True)
plt.show()


# ------Цепочка пропущенных строк
# 1. Список всех файлов
# path = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\*.csv" 
# files = glob.glob(path)

# overall_max_gap = 0
# file_with_max_gap = ""

# print(f"Начинаю поиск в {len(files)} файлах...")

# for file in files:
#     # Читаем только нужные колонки, чтобы было быстрее (например, 'vibration')
#     # Если пропуски могут быть в разных колонках, лучше проверить все
#     df = pd.read_csv(file) 
    
#     for column in df.columns:
#         # Создаем маску пропусков (True там, где NaN)
#         is_null = df[column].isnull()
        
#         if is_null.any():
#             # Хитрая математика: группируем идущие подряд True
#             # diff() ловит моменты изменения (был пропуск - стал нет, и наоборот)
#             # cumsum() создает уникальный номер для каждой такой группы
#             groups = (is_null != is_null.shift()).cumsum()
            
#             # Считаем размер каждой группы, где были пропуски
#             gaps = is_null.groupby(groups).sum()
#             current_max = gaps.max()
            
#             if current_max > overall_max_gap:
#                 overall_max_gap = current_max
#                 file_with_max_gap = os.path.basename(file)

# print("-" * 30)
# print(f"Самая длинная цепочка пропусков: {int(overall_max_gap)} строк")
# print(f"Она находится в файле: {file_with_max_gap}")
