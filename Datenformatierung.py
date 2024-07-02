import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# Funktion zum Entfernen von nicht numerischen Zeichen
def remove_non_numeric(s):
    return ''.join([char for char in s if char.isdigit()])

# Daten einlesen
df = pd.read_csv('estat_birth_1960.csv', delimiter=';')
# Einzeldaten splitten
df_split = df['freq,unit,month,geo'].str.split(',', expand=True)
# Datendefinition splitten
df.insert(0, 'GEO', df_split[3])
df.insert(0, 'MONTH', df_split[2])
df.insert(0, 'UNIT', df_split[1])
df.insert(0, 'FREQ', df_split[0])

# Spalten löschen
df = df.drop(['freq,unit,month,geo'], axis=1)

df.to_csv('estat_birth_1960_clean.csv', index=False)

df_data_dict = {}

df_dumy = df[df['MONTH'] == 'TOTAL']

#print(df_dumy)

#Sichere die Daten jedere Spalte in einem Dictionary
for column in df_dumy.columns:
	df_data_dict[column] = df_dumy[column]

# Create a list for the timeline
df_data_dict['Timeline'] = []

for cnt in range(1960, 2023, 1):
	df_data_dict['Timeline'].append(cnt)


for key in range(0, 59, 1):
	geo_name = df_data_dict['GEO'].values[key]
	# Debug print
	#print(geo_name)
	# Create a list on this marker
	df_data_dict[geo_name] = []
	#print(df_data_dict[geo_name])
	# Interate through all years
	for cnt in df_data_dict['Timeline']:
		# Debug print
		#print(cnt)
		# Get the value of the timeline
		timeline_value = df_dumy[str(cnt)].values[key]
		# Remove the non numeric characters
		timeline_value = remove_non_numeric(timeline_value)
		# Check if the value is not a number
		if timeline_value == '':
			# Set the value to 0
			timeline_value = 0
		# make it an integer
		timeline_value = int(timeline_value)
		# Stick it in the list of the Country
		df_data_dict[geo_name].append(timeline_value)
		# Debug Print
		#print(df_data_dict[str(cnt)].values[key])

	# Debug print
	#print(df_data_dict[geo_name])

#print(df_data_dict)

#####################################################################
# The Data is now in the dictionary df_data_dict separated to the countries
#####################################################################

# Plot the data
#for country in df_data_dict['GEO'].values:
#	plt.plot(df_data_dict['Timeline'], df_data_dict[country], label=country)
#plt.title('Birthrate in Europe')
#plt.xlabel('Year')
#plt.ylabel('Birthrate')
#plt.legend()
#plt.show()


#countries_to_plot = ['DE', 'IT', 'AT', 'CH', 'FR', 'GB']

#for country in countries_to_plot:
#    if country in df_data_dict:
#        plt.plot(df_data_dict['Timeline'], df_data_dict[country], label=country)

#plt.title('Birthrate in European Countries from 1960 to 2022')
#plt.xlabel('Year')
#plt.ylabel('Birthrate')
#plt.legend()
#plt.savefig('Birthrate.png')
#plt.show()

########################################################################
# Birthrates for different countries stored in usable variables
########################################################################

df_total_births_DE = df_data_dict['DE']
df_total_births_AT = df_data_dict['AT']
df_total_births_IT = df_data_dict['IT']

df_Years = df_data_dict['Timeline']

total_deaths_AT = np.loadtxt('annual_deaths_total.csv', delimiter=',', unpack=True, skiprows=117, max_rows=62, usecols=(7))

df_total_deaths_AT = pd.DataFrame(total_deaths_AT)

population_AT = np.loadtxt('population_january_first.csv', delimiter = ',', unpack=True, skiprows=115, max_rows=62, usecols=(8))

df_population_AT = pd.DataFrame(population_AT)

df_data = pd.merge(df_Years, df_population_AT, df_total_births_AT, df_total_deaths_AT)

# Calculate birth and death rates per 1000 people


df_data['birth_rate'] = (df_data['df_total_births_AT'] / df_data['df_population']) * 1000
df_data['death_rate'] = (df_data['df_total_deaths_AT'] / df_data['df_population']) * 1000


# Population projection

df_data['predicted_population'] = df_data['df_population_AT']

for i in range(1,len(df_data)):
	df.loc[i, 'predicted_population'] = df.loc[i-1, 'predicted_population'] + (df.loc[i-1, 'Birth_Rate_AT'] - df.loc[i-1, 'Death_Rate_AT'])


plt.figure(figsize=(10, 6))
plt.plot(df_data['df_Years'], df_data['predicted_population'], marker='o', label='Predicted Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Projection')
plt.legend()
plt.grid(True)
plt.show()





