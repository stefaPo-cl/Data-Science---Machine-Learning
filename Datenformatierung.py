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

#df_total_births_DE = df_data_dict['DE']
#df_total_births_AT = df_data_dict['AT']
#df_total_births_IT = df_data_dict['IT']

#df_Years = df_data_dict['Timeline']


#df_total_deaths = pd.read_csv('annual_deaths_total.csv', delimiter=',', names=['Country', 'Year', 'Total_Deaths'], usecols=(5,6,7))

#total_deaths_AT = np.loadtxt('annual_deaths_total.csv', delimiter=',', unpack=True, skiprows=117, max_rows=62, usecols=(7))

#df_population = pd.read_csv('population_january_first.csv', delimiter=',', names=['Country', 'Year', 'Total_Deaths'], usecols=(6,7,8))

#population_AT = np.loadtxt('population_january_first.csv', delimiter = ',', unpack=True, skiprows=115, max_rows=62, usecols=(8))

#df_population_AT = df_population[df_population['Country'] == 'AT']

#df_total_deaths_AT = df_total_deaths[df_total_deaths['Country'] == 'AT']

#print(df_total_deaths_AT)
#print(df_population_AT)



# Read the CSV files into DataFrames
deaths_df = pd.read_csv('annual_deaths_total.csv', header=None, names=['country', 'year', 'deaths'], usecols=[5,6,7])
population_df = pd.read_csv('population_january_first.csv', header=None, names=['country', 'year', 'population'], usecols=[6,7,8])


# Assuming df_data_dict is the dictionary obtained from the cleaned birth data
# Extract the timeline
timeline = df_data_dict['Timeline']

# Initialize a list to store the merged data for all countries
austria_deaths = deaths_df[deaths_df['country'] == 'AT'][['year', 'deaths']]
austria_population = population_df[population_df['country'] == 'AT'][['year', 'population']]

# Extract Austria's birth data
if 'AT' in df_data_dict:
    austria_births = df_data_dict['AT']
else:
    # If birth data is missing for Austria, create a list of NaNs
    austria_births = [None] * len(timeline)

# Create a DataFrame for the birth data
austria_births_df = pd.DataFrame({
    'births': austria_births
})



# Merge the data for Austria
AT_deaths_pop_merged = pd.merge(austria_deaths, austria_population, on='year', how='outer')
austria_merged = AT_deaths_pop_merged.join(austria_births_df)
austria_merged['country'] = 'AT'

#print(AT_deaths_pop_merged)
#print(austria_births)
#print(austria_births_df)
#print(austria_merged)

# Handle missing values
austria_merged = austria_merged.dropna()  # Or use .fillna() to replace missing values

# Display the cleaned and merged DataFrame for Austria
#print(austria_merged.head())

# Save the merged DataFrame for Austria to a CSV file (optional)
austria_merged.to_csv('austria_merged_data.csv', index=False)
#print(df_data_AT)

#df_data = pd.merge(df_Years, df_population_AT, df_total_births_AT, df_total_deaths_AT)

# Calculate birth and death rates per 1000 people

df_data_AT = pd.read_csv('austria_merged_data.csv', header=None, names=['year', 'deaths', 'population', 'births', 'country'], skiprows=1)

#print(df_data_AT)

df_data_AT['birth_rate'] = (df_data_AT['births'] / df_data_AT['population']) * 1000
df_data_AT['death_rate'] = (df_data_AT['deaths'] / df_data_AT['population']) * 1000

# Initialize columns for predictions
df_data_AT['predicted_population'] = df_data_AT['population']
for i in range(1, len(df_data_AT)):
    df_data_AT.loc[i, 'predicted_population'] = df_data_AT.loc[i-1, 'predicted_population'] + (df_data_AT.loc[i-1, 'births'] - df_data_AT.loc[i-1, 'deaths'])

#print(df_data_AT.head())

plt.figure(figsize=(10, 6))
plt.plot(df_data_AT['year'], df_data_AT['population'], marker='o', label='Actual Population')
plt.plot(df_data_AT['year'], df_data_AT['predicted_population'], marker='o', label='Predicted Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Projection')
plt.legend()
plt.grid(True)
plt.show()




