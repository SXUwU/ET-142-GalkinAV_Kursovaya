import pandas as pd
import os

datalist = os.listdir("C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure")

names_for_column = ["Vibration_X", "Vibration_Y", "Temp_Bearing", "Temp_Atmosphere"]

framelist = []

path = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure" + "\\" + datalist[0]
framelist.append(pd.read_csv(path, names=names_for_column).describe())

path = "C:\\Users\\Aleks\\OneDrive\\Desktop\\Предметы\\Введение в проектную деятельность\\Курсовая\\Vibration_Bearing_RuntoFailure" + "\\" + datalist[len(datalist)-1]
framelist.append(pd.read_csv(path, names=names_for_column).describe())   



print(framelist[0])
print("-"*40)


print(framelist[1])
print("-"*40)