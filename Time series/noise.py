import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_bearing_decomposition(file_path):
    
    df = pd.read_csv(file_path, header=None, usecols=[2], names=['Temp'])
    
    # Усредние каждые 12800 точек (0.5 секунда), чтобы выделить тренд
    temp_series = df['Temp'].groupby(np.arange(len(df)) // 12800).mean()
    
    decomposition = seasonal_decompose(temp_series, model='additive', period=60)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    resid = decomposition.resid  
    
    # Выделение сигнала (Сигнал = Тренд + Сезонность)
    signal = trend + seasonal

    # Расчет SNR
    # Удаление NaN 
    mask = ~np.isnan(signal) & ~np.isnan(resid)
    var_signal = np.var(signal[mask])
    var_noise = np.var(resid[mask])
    
    # Формула: SNR = 10 * lg(σ² сигнал / σ² шум)
    snr = 10 * np.log10(var_signal / var_noise)

    # Визуализация декомпозиции
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(temp_series, label='Исходный ряд', color='black')
    plt.legend(loc='upper left')
    plt.title('Декомпозиция канала: Температура подшипника')

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Тренд', color='blue')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Сезонность', color='green')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 4)
    plt.scatter(range(len(resid)), resid, label='Остатки (Шум)', color='red', s=1)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

    # Гистограмма распределения шума
    plt.figure(figsize=(10, 5))
    sns.histplot(resid[mask], kde=True, color='red', bins=50)
    plt.title(f'Распределение остатков (Шума). SNR = {snr:.2f} dB')
    plt.xlabel('Амплитуда шума')
    plt.ylabel('Частота')
    plt.show()

    return snr


snr_value = analyze_bearing_decomposition('C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure\\LogFile_2022-06-26-01-00-31.csv')