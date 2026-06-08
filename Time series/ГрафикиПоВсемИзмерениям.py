import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Настройка пути к данным
# Замени на путь к твоей папке, где лежат все CSV файлы
DATA_DIR = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure" 

# Ищем все файлы .csv и сортируем их по имени, чтобы соблюсти хронологию
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

if not file_paths:
    print(f"Ошибка! В папке '{DATA_DIR}' не найдено файлов .csv. Проверь путь.")
    exit()

print(f"Найдено файлов для обработки: {len(file_paths)}")

# Списки, куда мы будем собирать метрики по каждому файлу
vibration_x_rms = []
vibration_x_max_min = []
vibration_y_rms = []
vibration_y_max_min = []
temp_bearing_max_min = []
temp_atm_max_min = []
bearing_temp_mean = []
atmo_temp_mean = []
time_indices = []



# 2. Цикл поочередного чтения файлов (экономим оперативку)
for idx, path in enumerate(file_paths):
    file_name = os.path.basename(path)
    print(f"[{idx + 1}/{len(file_paths)}] Обработка файла: {file_name}")
    
    try:
        # Читаем файл. Заголовков в исходниках нет, задаем имена колонок вручную
        df = pd.read_csv(
            path, 
            header=None, 
            names=['vib_x', 'vib_y', 'temp_bearing', 'temp_atmo']
        )
        
        # Вычисляем СКЗ (RMS) для каналов вибрации
        rms_x = np.sqrt(np.mean(df['vib_x'] ** 2))
        rms_y = np.sqrt(np.mean(df['vib_y'] ** 2))
        
        # Вычисляем среднюю температуру за этот промежуток времени
        mean_bearing = df['temp_bearing'].mean()
        mean_atmo = df['temp_atmo'].mean()
        
        # Сохраняем агрегированные данные в наши списки
        vibration_x_rms.append(rms_x)
        vibration_y_rms.append(rms_y)
        bearing_temp_mean.append(mean_bearing)
        atmo_temp_mean.append(mean_atmo)
        
        vibration_x_max_min.append(df['vib_x'].max())
        vibration_x_max_min.append(df['vib_x'].min())
        vibration_y_max_min.append(df['vib_y'].max())
        vibration_y_max_min.append(df['vib_y'].min())
        temp_bearing_max_min.append(df["temp_bearing"].max())
        temp_bearing_max_min.append(df['temp_bearing'].min())
        temp_atm_max_min.append(df["temp_atmo"].max())
        temp_atm_max_min.append(df["temp_atmo"].min())
        
        time_indices.append(idx)  # Шаг времени (номер файла)
        
    except Exception as e:
        print(f"Не удалось прочитать файл {file_name}. Ошибка: {e}")



print(f"По x: {max(vibration_x_max_min)} и {min(vibration_x_max_min)}")
print(f"По y: {max(vibration_y_max_min)} и {min(vibration_y_max_min)}")
print(f"По bearing: {max(temp_bearing_max_min)} и {min(temp_bearing_max_min)}")
print(f"По atmo: {max(temp_atm_max_min)} и {min(temp_atm_max_min)}")

# 3. Визуализация результатов (Графики в столбик)
# Создаем сетку: 4 графика друг под другом (4 строки, 1 столбец)
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# График 1: Вибрация X (RMS)
axs[0].plot(time_indices, vibration_x_rms, color='blue', linewidth=2, label='Vibration X (RMS)')
axs[0].set_ylabel('Ускорение (g)')
axs[0].set_title('Канал 1: Вибрация по оси X (Среднеквадратичное значение)')
axs[0].grid(True, linestyle='--')
axs[0].legend(loc='upper left')

# График 2: Вибрация Y (RMS)
axs[1].plot(time_indices, vibration_y_rms, color='orange', linewidth=2, label='Vibration Y (RMS)')
axs[1].set_ylabel('Ускорение (g)')
axs[1].set_title('Канал 2: Вибрация по оси Y (Среднеквадратичное значение)')
axs[1].grid(True, linestyle='--')
axs[1].legend(loc='upper left')

# График 3: Температура подшипника
axs[2].plot(time_indices, bearing_temp_mean, color='red', linewidth=2, label='Bearing Temp (Mean)')
axs[2].set_ylabel('Температура (°C)')
axs[2].set_title('Канал 3: Температура подшипника (Средняя за замер)')
axs[2].grid(True, linestyle='--')
axs[2].legend(loc='upper left')

# График 4: Температура атмосферы
axs[3].plot(time_indices, atmo_temp_mean, color='green', linewidth=2, label='Atmosphere Temp (Mean)')
axs[3].set_ylabel('Температура (°C)')
axs[3].set_title('Канал 4: Температура окружающей среды (Средняя за замер)')
axs[3].set_xlabel('Хронология эксперимента (Номер файла / Часы работы)')
axs[3].grid(True, linestyle='--')
axs[3].legend(loc='upper left')

# Оптимизируем расстояния между графиками, чтобы подписи не налезали друг на друга
plt.tight_layout()

plt.savefig('bearing_lifespan_trends.png', dpi=300)
plt.show()
print("Готово! Графики построены и сохранены как 'bearing_lifespan_trends.png'")