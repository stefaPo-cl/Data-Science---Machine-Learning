import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pygam import PoissonGAM, s, te, l
#from prophet import Prophet

###########################################################################################################################################
# City to analyse
Country = 'AT'
###########################################################################################################################################

# Function to call for non numeric characters
def remove_non_numeric(s):
    return ''.join([char for char in s if char.isdigit()])

# Load the data
df = pd.read_csv('Erste_Daten\estat_birth_1960.csv', delimiter=';')

####################################################################
# Data load to dataframe and first structuring
####################################################################
# The first column represents the definition of the data altogether, which we can not accept for proper data structuring
# Therefore we split the first column into four columns
df_split = df['freq,unit,month,geo'].str.split(',', expand=True)

# Insert the data in the single column in the single columns
df.insert(0, 'GEO', df_split[3])
df.insert(0, 'MONTH', df_split[2])
df.insert(0, 'UNIT', df_split[1])
df.insert(0, 'FREQ', df_split[0])

# Delete the single column
df = df.drop(['freq,unit,month,geo'], axis=1)

# Save the first formation sequence to a csv-file
df.to_csv('Formatiert\estat_birth_1960_clean.csv', index=False)

####################################################################
# Further data transformation for the Project
####################################################################

# Create a dicctionary for a refined data structure
df_data_dict = {}

# The last few entries are the total amounts of the births in ervery country every year
df_compl = df[df['MONTH'] == 'TOTAL']

# Debug print
#print(df_dumy)

# Save every column in the dictionary
for column in df_compl.columns:
	df_data_dict[column] = df_compl[column]

# Create a list for the timeline
df_data_dict['Timeline'] = []

# Our Timeline reaches from 1960 to 2023, therefore we create a Timeline
for cnt in range(1960, 2023, 1):
	df_data_dict['Timeline'].append(str(cnt))

# The CSV-file we got from the EUROSTAT-Website is for us wrong configurated (data per city in rows not in columns)
# Therefore we have to alter the data to get a dictionary for every country from 1960 to 2023
# We have exactly 59 Entries representing countries and further specific data
for key in range(0, 59, 1):
	# Get the name of every country/data-set
	geo_name = df_data_dict['GEO'].values[key]

	# Debug print
	#print(geo_name)
	
	# Create a list on this marker
	df_data_dict[geo_name] = []

	# Debug print
	#print(df_data_dict[geo_name])

	# Iterate through all years
	for cnt in df_data_dict['Timeline']:
		# Debug print
		#print(cnt)
		# Get the value of the timeline
		timeline_value = df_compl[str(cnt)].values[key]
		# Remove the non numeric characters
		timeline_value = remove_non_numeric(timeline_value)
		# Check if the value is not a number
		if timeline_value == '':
			# Set the value to 0
			timeline_value = 0
		# make it an integer
		timeline_value = str(timeline_value)
		# Stick it in the list of the Country
		df_data_dict[geo_name].append(timeline_value)
		# Debug Print
		#print(df_data_dict[str(cnt)].values[key])

	# Debug print for all the data depending on the country
	#print(df_data_dict[geo_name])

# Debug print for the dictionary where all of the data is stored
#print(df_data_dict)

# The birthrate data is now formated as we want it to be
#####################################################################
# Further data Analysis
#####################################################################
# Next we need to download the next datasets we want to include
# Therefore we found:
# Annual_deaths_total.csv
# migration_data.csv
# population.csv measured every first day of the year

# Import the selected data
deaths_df 		= pd.read_csv('Erste_Daten/annual_deaths_total.csv', 		header=None, names=['country', 'year', 'deaths'], 		usecols=[5,6,7])
population_df 	= pd.read_csv('Erste_Daten/population_january_first.csv', 	header=None, names=['country', 'year', 'population'],	usecols=[6,7,8])
imigration_df	= pd.read_csv('Erste_Daten/migr_imm8_linear_2_0.csv', 		header=None, names=['age', 'sex', 'country', 'year', 'imigrants'], usecols=[4, 6, 7, 8, 9], low_memory=False)
emigration_df 	= pd.read_csv('Erste_Daten/emigration.csv', 				header=None, names=['country', 'year', 'emigrants'], 		usecols=[7, 8, 9])

# debug print
#print(imigration_df)
#######################################
# Data structuring of the imigration_data
#######################################
# It looks like the data is separated through sex: Male, Female, Trans
# To look at the complete data we have to sum up all of the total data

# Create imigration dictionary
imigration_dict = {}

# Extract the Total data for all countries
imigration_total = imigration_df[imigration_df['age'] == 'TOTAL']
imigration_total = imigration_total[imigration_total['sex'] == 'T']

# In the dataset, the total data is saved twice therefore we have to eliminate the second savings
imigration_total = imigration_total[imigration_total.index < 2780]

# Debug print
#print(imigration_total[imigration_total['country'] == 'DE']['year'])

# To plot the data we have to reassign the data to integers
#imigration_total['year'] = imigration_total['year'].astype(int)
#imigration_total['imigrants'] = imigration_total['imigrants'].astype(int)

# Debug print
#print(imigration_total[imigration_total['country'] == 'DE'])

## Plot the imigration data of Germany and Austria
#plt.plot(imigration_total[imigration_total['country'] == 'DE']['year'], imigration_total[imigration_total['country'] == 'DE']['imigrants'], label='Imigration in Germany')
#plt.plot(imigration_total[imigration_total['country'] == 'AT']['year'], imigration_total[imigration_total['country'] == 'AT']['imigrants'], label='Imigration in Austria')
#plt.legend()
#plt.xlabel('Year')
#plt.ylabel('Imigrants')
#plt.title('Imigration in Germany and Austria')
#plt.show()

#######################################
# Data analysis for a Country specified in the beginning
#######################################
# We want the data for Austria to be structured in a way we can use it for our project
# Therefore we have to merge the data of the birthrate, the deathrate, imigration and the population

# Extract the data for the country mentioned above and if the data is not complete we have to fill it with NaN
try:
	country_deaths_df = deaths_df[deaths_df['country'] == Country][['year', 'deaths']]
except Exception as e:
	print('Country not found in the dataset')
	country_deaths_df = pd.DataFrame({
		'year': df_data_dict['Timeline'],
		'deaths': [None] * len(df_data_dict['Timeline'])
	})

try:
	country_population_df = population_df[population_df['country'] == Country][['year', 'population']]
except Exception as e:
	print('Country not found in the dataset')
	country_population_df = pd.DataFrame({
		'year': df_data_dict['Timeline'],
		'population': [None] * len(df_data_dict['Timeline'])
	})

try:
	country_imigration_df = imigration_total[imigration_total['country'] == Country][['year', 'imigrants']]
except Exception as e:
	print('Country not found in the dataset')
	country_imigration_df = pd.DataFrame({
		'year': df_data_dict['Timeline'],
		'imigrants': [None] * len(df_data_dict['Timeline'])
	})

try:
	country_emigration_df = emigration_df[emigration_df['country'] == Country][['year', 'emigrants']]
except Exception as e:
	print('Country not found in the dataset')
	country_emigration_df = pd.DataFrame({
		'year': df_data_dict['Timeline'],
		'emigrants': [None] * len(df_data_dict['Timeline'])
	})

# Birth data is already sorted from 1960 to 2023
country_births_df = df_data_dict[Country]

# Check if every dataset is same in length
if len(country_deaths_df) != len(country_births_df) or len(country_population_df) != len(country_births_df) or len(country_imigration_df) != len(country_births_df) or len(country_emigration_df) != len(country_births_df):
	print('Datasets are not the same length')
else:
	print('Datasets are the same length')

# Create a DataFrame for the birth data
country_births_df = pd.DataFrame({
	'year': df_data_dict['Timeline'],
	'births': country_births_df
})

#print(country_births_df)
#print(country_deaths_df)

# Merge the data for the country
country_deaths_pop_merged = pd.merge(country_deaths_df, country_population_df, on='year', how='outer')
country_three_pop_merged = pd.merge(country_deaths_pop_merged, country_imigration_df, on='year', how='outer')
country_four_pop_merged = pd.merge(country_three_pop_merged, country_emigration_df, on='year', how='outer')
country_merged = pd.merge(country_four_pop_merged, country_births_df, on='year', how='outer')


#Debug print
#print(country_merged)

# Delete Line 2023 because there is not much data given
country_merged = country_merged[country_merged['year'] != '2023']

country_merged.fillna('0', inplace=True)

country_merged.to_csv('Merged_Data/' + Country + '_merged_data.csv', index=False)

##########################################
# Data analysis for the country specified
##########################################

## Plot for the population
#plt.plot(country_merged['year'].astype(int), country_merged['population'].astype(int), marker='o', label='Population')
#plt.xlabel('Year')
#plt.ylabel('Different Data')
#plt.title('Different Data for ' + Country)
#plt.grid(True)
#plt.legend()
#plt.show()
#
## Plot for the birthrate, deathrate and imigration
#plt.plot(country_merged['year'].astype(int), country_merged['births'].astype(int), marker='o', label='Births')
#plt.plot(country_merged['year'].astype(int), country_merged['deaths'].astype(int), marker='o', label='Deaths')
#plt.plot(country_merged['year'].astype(int), country_merged['imigrants'].astype(int), marker='o', label='Imigrants')
#plt.plot(country_merged['year'].astype(int), country_merged['emigrants'].astype(int), marker='o', label='Emigrants')
#plt.xlabel('Year')
#plt.ylabel('Different Data')
#plt.title('Different Data for ' + Country)
#plt.grid(True)
#plt.legend()
#plt.show()

##################################################
# Linear Regression for Prediction in the past
##################################################
ds_header = ['imigrants', 'emigrants', 'deaths', 'births', 'population']

for header in ds_header:
	for i in range(len(country_merged['year'])):
		if country_merged[header].values[i] != '0':
			X = country_merged['year'].astype(int).values[i:]
			y = country_merged[header].astype(int).values[i:]
			break

	X = X.reshape(-1, 1)

	# Check if the dataset is complete
	if len(X) != len(country_merged['year']):
		poly = PolynomialFeatures(degree=5)
		x_poly = poly.fit_transform(X)

		model = LinearRegression()
		model.fit(x_poly, y)

		# Vorhersage in die Vergangenheit
		years_past = np.arange(1960, X.min()).reshape(-1, 1)
		years_past_poly = poly.fit_transform(years_past)
		predictions_past = model.predict(years_past_poly)

		## Plot die Vorhersage in die Vergangenheit
		#plt.plot(years_past, predictions_past, label='Predictions Past ' + header)
		#plt.plot(X, y, label='Actual Data ' + header)
		#plt.xlabel('Year')
		#plt.ylabel('Different Data')
		#plt.title('Different Data for ' + Country)
		#plt.grid(True)
		#plt.legend()
		#plt.show()
	else:
		print('Dataset ' + header + ' is complete')

##################################################
# Linear Regression for Prediction in the future
##################################################
# We want to predict the future population of the country mentioned above
# Therefore we have to find coefficients which describe each factor altering the population
# Birthrate, Deathrate, Imigrationrate, Emigrationrate
# We find the coefficients per 1000 people

# Create Variables for each factor
country_merged['birthrate'] = (country_merged['births'].astype(int) / country_merged['population'].astype(int)) * 1000
country_merged['deathrate'] = (country_merged['deaths'].astype(int) / country_merged['population'].astype(int)) * 1000
country_merged['imigrationrate'] = (country_merged['imigrants'].astype(int) / country_merged['population'].astype(int)) * 1000
country_merged['emigrationrate'] = (country_merged['emigrants'].astype(int) / country_merged['population'].astype(int)) * 1000

# DataFrame for the coefficients
coefficients = pd.DataFrame({
	'birthrate' : [],
	'deathrate' : [],
	'imigrationrate' : [],
	'emigrationrate' : []
})

# In the calculation into the future deaths and emigrants are negative, thereofore we have to change the sign
country_merged = country_merged.astype(float)

# Define emigrants and deaths as negative
country_merged['emigrants'] = -country_merged['emigrants']
country_merged['deaths'] = -country_merged['deaths']

country_merged = country_merged.astype(str)


# Save everything in a csv-file
country_merged.to_csv('Merged_Data/' + Country + '_merged_data.csv', index=False)


# Alter the type of the data so that we can calculate with it
country_merged = country_merged.astype(float)

# Calculate the coefficients
coefficients['birthrate'] = country_merged['birthrate'].mean()
coefficients['deathrate'] = country_merged['deathrate'].mean()
coefficients['imigrationrate'] = country_merged['imigrationrate'].mean()
coefficients['emigrationrate'] = country_merged['emigrationrate'].mean()


# Build a population prediction database
country_merged['predicted_population'] = country_merged['population']

# Calculate the population for the future with the previous data
for i in range(36, len(country_merged)):
    country_merged.loc[i, 'predicted_population'] = country_merged.loc[i-1, 'predicted_population'] + (country_merged.loc[i-1, 'births'] + country_merged.loc[i-1, 'deaths'] + country_merged.loc[i-1, 'imigrants'] + country_merged.loc[i-1, 'emigrants'])

# Debug print
#print(df_data_AT.head())

## Plot the built population prediction
#plt.figure(figsize=(10, 6))
#plt.plot(country_merged['year'], country_merged['population'], marker='o', label='Actual Population')
#plt.plot(country_merged['year'], country_merged['predicted_population'], marker='o', label='Predicted Population')
#plt.xlabel('Year')
#plt.ylabel('Population')
#plt.title('Population Projection')
#plt.legend()
#plt.grid(True)
#plt.show()

# Generate a future prediction timeline
country_merged['year'] = country_merged['year'].astype(int)

country_merged['Future_year'] = country_merged['year']

for i in range(0, len(country_merged['year'])):
	country_merged.loc[i, 'Future_year'] = country_merged.loc[i, 'year'] + len(country_merged['year']) - 1


# Calculate the future population
country_merged['Future'] = None 

country_merged.loc[0, 'Future'] = country_merged.loc[62, 'predicted_population']

print(country_merged)

for i in range(1, len(country_merged)):
    country_merged.loc[i, 'Future'] = country_merged.loc[i-1, 'Future'] + (country_merged.loc[i-1, 'births'] + country_merged.loc[i-1, 'deaths'] + country_merged.loc[i-1, 'imigrants'] + country_merged.loc[i-1, 'emigrants'])


print(country_merged)

# Plot the built population prediction
plt.figure(figsize=(10, 6))
plt.plot(country_merged['year'], country_merged['population'], marker='o', label='Actual Population')
plt.plot(country_merged['Future_year'], country_merged['Future'], marker='o', label='Predicted Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Projection')
plt.legend()
plt.grid(True)
plt.show()

## Look into the future with a GAM model
#X = country_merged[country_merged['year'] >= 1996][['year', 'deaths', 'births', 'imigrants', 'emigrants']]
#y = country_merged[country_merged['year'] >= 1996]['population'].to_numpy()
#
#gam = PoissonGAM(s(0) + te(1, 2)).fit(X, y)
#
#lam = np.logspace(-3, 3, 10)
#gam.gridsearch(X, y, lam=lam)
#
## Predict future values
#years_future = np.arange(2023, 2033).reshape(-1, 1)
#X_future = np.hstack([years_future, np.tile(X.mean(axis=0)[1:], (len(years_future), 1))])  # Assume mean of other features for future
#y_pred_future = gam.predict(X_future)
#
#plt.plot(country_merged[country_merged['year'] >= 1996]['year'], y, 'o', label='Actual Population')
#plt.plot(X_future[:, 0], y_pred_future, 'x', label='Predicted Population', linestyle='dashed')
#plt.xlabel('Year')
#plt.ylabel('Population')
#plt.title('Population Projection with Poisson GAM')
#plt.legend()
#plt.grid(True)
#plt.show()