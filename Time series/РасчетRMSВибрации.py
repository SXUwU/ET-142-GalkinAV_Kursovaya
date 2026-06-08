import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Укажи путь к папке, где лежат твои 129 файлов
DATA_FOLDER = 'C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure' 

# Собираем все .csv файлы и сортируем их, чтобы идти по порядку
files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
files.sort() # Важно: сортировка гарантирует, что мы идем от 1 до 129 файла

rms_x_history = []
rms_y_history = []

print(f"Найдено файлов: {len(files)}")

for file in files:
    try:
        # Читаем только нужные колонки: X, Y (индексы 0 и 1)
        # Это сэкономит память, так как не читаем температуру
        df = pd.read_csv(file, header=None, usecols=[0, 1])
        
        # Считаем RMS (энергию)
        rms_x = np.sqrt(np.mean(df[0]**2))
        rms_y = np.sqrt(np.mean(df[1]**2))
        
        rms_x_history.append(rms_x)
        rms_y_history.append(rms_y)
    except Exception as e:
        print(f"Ошибка в файле {file}: {e}")
        rms_x_history.append(None)
        rms_y_history.append(None)

# --- Визуализация ---
plt.figure(figsize=(12, 6))
plt.plot(rms_x_history, label='RMS Vibration X', color='blue', alpha=0.7)
plt.plot(rms_y_history, label='RMS Vibration Y', color='green', alpha=0.7)

plt.title('График деградации: Рост уровня вибрации (RMS) по всей серии')
plt.xlabel('Порядковый номер файла в папке')
plt.ylabel('RMS (Vibration Level)')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()