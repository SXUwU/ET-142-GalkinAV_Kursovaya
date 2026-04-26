import pandas as pd
import os
import dask.dataframe as dd

def get_global_statistics(folder_path):
    column_names = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]
    
    # 1. Dask читает ВСЕ файлы по маске *.csv как единый датасет
    df = dd.read_csv(f'{folder_path}/*.csv', header=None, names=column_names)
    
    stats_dask = df.describe()
    
    # compute() запускает реальный процесс чтения файлов и расчетов
    global_stats = stats_dask.compute()
    
    global_stats = global_stats.rename(index={
        '25%': 'Q1',
        '50%': 'Q2 (Median)',
        '75%': 'Q3'
    })
    
    return global_stats


final_stats = get_global_statistics('C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure')
print(final_stats)



# datalist = os.listdir("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure")

# names_for_column = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]

# framelist = []


# for i in range(len(datalist)):
#     path = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure" + "\\" + datalist[i]
#     framelist.append(pd.read_csv(path, names=names_for_column).describe())
#     break

# print(framelist[0])

    
