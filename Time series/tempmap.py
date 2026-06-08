import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_analysis(folder_path):
    column_names = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    sampled_data = []

    # Собор данных (каждая 1000-я строка)
    for file_path in all_files:
        df = pd.read_csv(file_path, header=None, names=column_names)
        sampled_data.append(df.iloc[::1000, :])
    
    full_df = pd.concat(sampled_data, ignore_index=True)

    # 2. Вычисление матрицы корреляции Пирсона
    corr_matrix = full_df.corr(method='pearson')

    # 3. Визуализация в виде тепловой карты (Heatmap)
    plt.figure(figsize=(10, 8))
   
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    plt.title("Тепловая карта корреляции каналов (Pearson Correlation)", fontsize=15)
    plt.tight_layout()
    plt.show()

    return corr_matrix

correlation_matrix = correlation_analysis('C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure')

print(correlation_matrix)