import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# def analyze_all_files(folder_path):
#     # 1. Подготовка
#     column_names = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]
#     # Ищем все файлы .csv в папке
#     all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
#     # Список для хранения результатов по каждому файлу
#     results = []


#     # 2. Цикл по всем файлам
#     for file_path in all_files:
#         file_name = os.path.basename(file_path)
#         df = pd.read_csv(file_path, header=None, names=column_names)
        
#         # Словарь для хранения метрик текущего файла
#         file_metrics = {"file": file_name}
        
#         for col in column_names:
#             # --- Расчет пропусков ---
#             missing_pct = df[col].isnull().mean() * 100
#             file_metrics[f"{col}_missing_%"] = missing_pct
            
#             # --- Расчет выбросов (3 сигмы) ---
#             mean = df[col].mean()
#             std = df[col].std()
#             lower = mean - 3 * std
#             upper = mean + 3 * std
            
#             outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
#             file_metrics[f"{col}_outliers_count"] = outliers_count

#         results.append(file_metrics)

#     # 3. Создание итоговой таблицы
#     summary_df = pd.DataFrame(results)
    
#     return summary_df


report = pd.read_csv('C:\\Users\\Aleks\\AppData\\Local\\Programs\\Microsoft VS Code\\dataset_anomaly_report.csv')

print(report["Vibration_X_outliers_count"].sum())
print(report["Vibration_Y_outliers_count"].sum())
print(report["Temp_Bearing_outliers_count"].sum())
print(report["Temp_Atmosphere_outliers_count"].sum())

plt.figure(figsize=(15, 6))

# Рисуем график роста количества выбросов по вибрации X
plt.plot(range(len(report)), report['Vibration_X_outliers_count'], label='Vibration X Outliers', color='blue')
plt.plot(range(len(report)), report['Vibration_Y_outliers_count'], label='Vibration Y Outliers', color='red')
plt.plot(range(len(report)), report['Temp_Bearing_outliers_count'], label='Temp Bearing outliers count', color='pink')

plt.title("Динамика количества выбросов за весь период эксперимента (129 часов)", fontsize=14)
plt.xlabel("Время (часы)")
plt.ylabel("Количество выбросов (3-sigma)")
plt.legend()
plt.grid(True)
plt.show()