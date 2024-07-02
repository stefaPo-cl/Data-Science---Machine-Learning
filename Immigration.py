import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# Funktion zum Entfernen von nicht numerischen Zeichen
#def remove_non_numeric(s):
#    return ''.join([char for char in s if char.isdigit()])

# Daten einlesen
#df = pd.read_csv('migr_imm8_linear_2_0.csv', delimiter=',')
# Einzeldaten splitten
#df_split = df['STRUCTURE,STRUCTURE_ID,freq,agedef,age,unit,sex,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG'].str.split(',', expand=True)
# Datendefinition splitten
#df.insert(0, 'OBS_FLAG', df_split[10])
#df.insert(0, 'OBS_VALUE', df_split[9])
#df.insert(0, 'TIME_PERIOD', df_split[8])
#df.insert(0, 'geo', df_split[7])
#df.insert(0, 'sex', df_split[6])
#df.insert(0, 'unit', df_split[5])
#df.insert(0, 'age', df_split[4])
#df.insert(0, 'agedef', df_split[3])
#df.insert(0, 'freq', df_split[2])
#df.insert(0, 'STRUCTURE_ID', df_split[1])
#df.insert(0, 'STRUCTURE', df_split[0])

## Spalten löschen
#df = df.drop(['STRUCTURE,STRUCTURE_ID,freq,agedef,unit,OBS_FLAG'], axis=1)
#
#df.to_csv('migr_immigr_clean.csv', index=False)
#
#df_data_dict = {}
#
#df_dumy = df[df['age'] == 'TOTAL']
#
#print(df_dumy)


time_period, immigrants = np.loadtxt('migr_imm8_linear_2_0.csv', delimiter=',', unpack=True, skiprows=1819, max_rows=27, usecols=(8, 9))

def format_func(value, tick_number):
    return f'{value / 1e5:.1f}'


plt.plot(time_period, immigrants)
plt.title('Total immigrants over the years in AT')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.gca().annotate('1e5', xy=(0, 1), xycoords='axes fraction', fontsize=10.5, ha='left', va='bottom', annotation_clip=False)
plt.tight_layout()
plt.savefig('Immigrants_in_AT.png')
plt.show()

