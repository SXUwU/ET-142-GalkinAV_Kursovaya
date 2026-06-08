import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# Отключаем предупреждения pandas для чистоты вывода
warnings.filterwarnings("ignore")

# --- Настройки ---
# Впиши сюда пути к своему первому и последнему файлам
FILE_FIRST_PATH = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\LogFile_2022-06-20-17-00-31.csv" 
FILE_LAST_PATH = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\LogFile_2022-06-26-01-00-31.csv"

# Шаг прореживания: берем каждую 2000-ю строку (из 2 000 000 получится 1000 точек)
STEP = 2000 
# Период для декомпозиции (подбирается экспериментально, 50 - хорошая стартовая точка для 1000 наблюдений)
DECOMP_PERIOD = 100

# Имена колонок согласно твоему описанию
COLUMNS = ['Vib_X', 'Vib_Y', 'Temp_Bearing', 'Temp_Atm']

def load_and_sample_data(filepath, step, cols):
    """Считывает csv без заголовков и прореживает данные."""
    # Читаем файл, сразу задавая имена колонкам
    df = pd.read_csv(filepath, header=None, names=cols)
    # Берем данные с заданным шагом и сбрасываем индексы
    df_sampled = df.iloc[::step].reset_index(drop=True)
    return df_sampled


def calculate_snr_stable(signal):
    """
    Стабильный расчет: SNR как отношение дисперсии сигнала к дисперсии шума.
    Используем метод простого вычитания тренда (через полином), 
    чтобы избежать NaN на краях.
    """
    # 1. Удаляем тренд через полиномиальную аппроксимацию (без NaN)
    x = np.arange(len(signal))
    poly = np.polyfit(x, signal, deg=2) # Аппроксимируем тренд параболой
    trend = np.polyval(poly, x)
    
    # 2. Шум — это то, что осталось после удаления тренда
    noise = signal - trend
    
    # 3. SNR как отношение мощности сигнала (без тренда) к мощности шума
    # В данном случае сигнал — это вариативность процесса
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    
    return 10 * np.log10(signal_power / noise_power)

def plot_decomposition(decomp, title_prefix, snr_value):
    """Отрисовка результатов декомпозиции."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    fig.suptitle(f'{title_prefix} | SNR: {snr_value:.2f} dB', fontsize=16)
    
    axes[0].plot(decomp.observed, color='blue')
    axes[0].set_ylabel('Observed')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(decomp.trend, color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[2].plot(decomp.seasonal, color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    axes[3].plot(decomp.resid, color='red', marker='o', linestyle='none', markersize=3)
    axes[3].set_ylabel('Residual (Noise)')
    axes[3].set_xlabel('Sample Index (Time Step)')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# --- Выполнение ---

# 1. Загрузка и прореживание
print("Загрузка первого файла...")
df_first = load_and_sample_data(FILE_FIRST_PATH, STEP, COLUMNS)

print("Загрузка последнего файла...")
df_last = load_and_sample_data(FILE_LAST_PATH, STEP, COLUMNS)

# 2. Декомпозиция (используем аддитивную модель, так как температура обычно не имеет экспоненциального роста колебаний)
print("Проведение декомпозиции...")
decomp_first = seasonal_decompose(df_first['Temp_Bearing'], model='additive', period=DECOMP_PERIOD)
decomp_last = seasonal_decompose(df_last['Temp_Bearing'], model='additive', period=DECOMP_PERIOD)

# 3. Расчет SNR
snr_first = calculate_snr_stable(df_first['Temp_Bearing'])
snr_last = calculate_snr_stable(df_last['Temp_Bearing'])

print(f"SNR (Исправный подшипник): {snr_first:.2f} dB")
print(f"SNR (Сломанный подшипник): {snr_last:.2f} dB")

# 4. Визуализация
plot_decomposition(decomp_first, "Декомпозиция температуры: Файл 1 (Исправный)", snr_first)
plot_decomposition(decomp_last, "Декомпозиция температуры: Файл 129 (Поломка)", snr_last)