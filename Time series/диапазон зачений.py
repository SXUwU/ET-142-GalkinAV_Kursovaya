import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_global_boxplot(folder_path):
    column_names = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    sampled_data = []

    print(f"Начало сбора данных из {len(all_files)} файлов...")

    for file_path in all_files:
        # Читение файла, выбор каждой 1000-й строки
        df = pd.read_csv(file_path, header=None, names=column_names)
        
        # Берем выборку (каждое 1000-е значение)
        df_sampled = df.iloc[::1000, :]
        sampled_data.append(df_sampled)
        
    # Объединие всех выборок в один DataFrame
    full_samples_df = pd.concat(sampled_data, ignore_index=True)
    
    print(f"Сбор завершен. Итого строк для анализа: {len(full_samples_df)}")

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    ax = sns.boxplot(data=full_samples_df, palette="husl", fliersize=1)
    
    plt.title("Общий диапазон значений по всем 129 часам работы", fontsize=16)
    plt.ylabel("Значение (Амплитуда / Градусы Цельсия)", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


plot_global_boxplot('C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure')