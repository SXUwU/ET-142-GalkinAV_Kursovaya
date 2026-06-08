import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

# --- Настройки ---
FILE_FIRST_PATH = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\LogFile_2022-06-20-17-00-31.csv" 
FILE_LAST_PATH = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\LogFile_2022-06-26-01-00-31.csv"


STEP = 4000
DECOMP_PERIOD = 100
COLUMNS = ['Vib_X', 'Vib_Y', 'Temp_Bearing', 'Temp_Atm']

def load_and_sample_data(filepath, step, cols):
    """Считывает данные и делает прореживание."""
    df = pd.read_csv(filepath, header=None, names=cols)
    return df.iloc[::step].reset_index(drop=True)

def calculate_stable_snr(signal, resid):
    """Вычисляет стабильный SNR на основе вариативности сигнала и шума."""
    mask = ~np.isnan(signal) & ~np.isnan(resid)
    
    # Используем дисперсию (var), так как вибрация колеблется относительно нуля
    var_signal = np.var(signal[mask])
    var_noise = np.var(resid[mask])
    
    if var_noise == 0:
        return float('inf')
    
    return 10 * np.log10(var_signal / var_noise)

def plot_vibration_decomposition(decomp_first, decomp_last, axis_name):
    """ Строит сравнительные графики декомпозиции для исправного и сломанного состояния. """
    fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex='col')
    
    fig.suptitle(f'Анализ декомпозиции вибрации по оси {axis_name}', fontsize=16, fontweight='bold')
    
    # --- Левая колонка: Исправный подшипник ---
    axes[0, 0].set_title(f'Файл 1 (Исправный)', fontsize=12)
    axes[0, 0].plot(decomp_first.observed, color='blue')
    axes[0, 0].set_ylabel('Observed')
    
    axes[1, 0].plot(decomp_first.trend, color='orange')
    axes[1, 0].set_ylabel('Trend')
    
    axes[2, 0].plot(decomp_first.seasonal, color='green')
    axes[2, 0].set_ylabel('Seasonal')
    
    axes[3, 0].plot(decomp_first.resid, color='red', marker='o', linestyle='none', markersize=2)
    axes[3, 0].set_ylabel('Residual (Noise)')
    axes[3, 0].set_xlabel('Sample Index')

    # --- Правая колонка: Сломанный подшипник ---
    axes[0, 1].set_title(f'Файл 129 (Поломка)', fontsize=12)
    axes[0, 1].plot(decomp_last.observed, color='darkblue')
    
    axes[1, 1].plot(decomp_last.trend, color='darkorange')
    
    axes[2, 1].plot(decomp_last.seasonal, color='darkgreen')
    
    axes[3, 1].plot(decomp_last.resid, color='darkred', marker='o', linestyle='none', markersize=2)
    axes[3, 1].set_xlabel('Sample Index')
    
    # Сетка для всех графиков
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()

# --- Выполнение ---

print("Загрузка данных...")
df_first = load_and_sample_data(FILE_FIRST_PATH, STEP, COLUMNS)
df_last = load_and_sample_data(FILE_LAST_PATH, STEP, COLUMNS)

# Проходим циклом по осям X и Y
for axis in ['Vib_X', 'Vib_Y']:
    print(f"Анализ компоненты {axis}...")
    
    # Декомпозиция
    decomp_first = seasonal_decompose(df_first[axis], model='additive', period=DECOMP_PERIOD)
    decomp_last = seasonal_decompose(df_last[axis], model='additive', period=DECOMP_PERIOD)
    
  
    # Отрисовка (выведет сравнительный график для текущей оси)
    plot_vibration_decomposition(decomp_first, decomp_last, axis.split('_')[1])
    

# Предположим, vib_x_first и vib_x_last — это твои массивы вибрации по X
rms_first = np.sqrt(np.mean(df_first['Vib_X']**2))
rms_last = np.sqrt(np.mean(df_last['Vib_X']**2))

print(f"RMS (Энергия)  | До: {rms_first:.4f} -> После: {rms_last:.4f} (Рост в {rms_last/rms_first:.1f} раз!)")
# -----------
rms_first = np.sqrt(np.mean(df_first['Vib_Y']**2))
rms_last = np.sqrt(np.mean(df_last['Vib_Y']**2))

print(f"RMS (Энергия)  | До: {rms_first:.4f} -> После: {rms_last:.4f} (Рост в {rms_last/rms_first:.1f} раз!)")