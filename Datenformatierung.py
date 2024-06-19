import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

print(df_data_dict)

#####################################################################
# The Data is now in the dictionary df_data_dict separated to the countries
#####################################################################

# Plot the data
for country in df_data_dict['GEO'].values:
	plt.plot(df_data_dict['Timeline'], df_data_dict[country], label=country)
plt.title('Birthrate in Europe')
plt.xlabel('Year')
plt.ylabel('Birthrate')
plt.legend()
plt.show()
