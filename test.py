import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read in the profiles dataset
profiles = pd.read_excel('profiles.xlsx')

# Read in the matches dataset
matches = pd.read_excel('matches.xlsx')
matches['Match'] = 1
# Preprocess the data
# perform one-hot encoding for Services Offered
services = profiles['Services Offered'].str.get_dummies(sep=', ')
services = services.add_prefix('services_')

# perform one-hot encoding for Geographical operation
geo = profiles['Geographical operation'].str.get_dummies(sep=', ')
geo = geo.add_prefix('geo_')
# perform one-hot encoding for Services Required
services_req = profiles['Services Required'].str.get_dummies(sep=', ')
services_req = services_req.add_prefix('services_req_')

profiles_encoded = pd.concat([
    profiles.drop(
        ['Services Offered', 'Geographical operation', 'Services Required'],
        axis=1), services, geo, services_req
],
                             axis=1)

# Create all possible combinations of profiles using the cartesian product
combinations = pd.merge(profiles_encoded.assign(key=1), profiles_encoded.assign(key=1), on='key').drop('key', axis=1)
# Filter out self-matches and reverse duplicates
combinations = combinations[combinations['Name_x'] != combinations['Name_y']]
combinations = combinations[combinations['Name_x'] < combinations['Name_y']]

# Merge the matches data with the combinations data to get a new column indicating if the pair is a match
combinations = pd.merge(combinations, matches, how='left', left_on=['Name_x', 'Name_y'], right_on=['P1 Name', 'P2 Name'])
combinations['Match'] = ~combinations['P1 Name'].isna()
combinations = combinations.drop(['Agenda Id', 'Meeting Number', 'P1 Name', 'P1 Confirmation Number', 'P2 Name', 'P2 Confirmation Number'], axis=1)
#combinations.to_excel("merged2.xlsx", index=False)

geo_cols = [col for col in combinations.columns if col.startswith('geo_')]
req_cols = [col for col in combinations.columns if col.startswith('services_req_')]
off_cols = [col for col in combinations.columns if col.startswith('services_')]

#  Define the input features and target variable
X = combinations[geo_cols + req_cols + off_cols]
y = combinations['Match']

# # Train a random forest classification model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Print the feature importances
importances = pd.Series(rf.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

# Sort the features by importance
feature_importances = feature_importances.sort_values('Importance',
                                                      ascending=False)

#feature_importances.to_excel("features_importance.xlsx", index=False)
# Standardize the data
scaler = StandardScaler()
combinations = combinations.drop(['Confirmation Number_x', 'Name_x', 'Confirmation Number_y', 'Name_y'], axis=1)
data_scaled = scaler.fit_transform(combinations)

# Create a PCA object
pca = PCA()

# Fit the PCA model on the data
pca.fit(data_scaled)

# Transform the data to the new space
data_transformed = pca.transform(data_scaled)

# Print the explained variance ratio for each principal component
print('Explained variance ratio:', pca.explained_variance_ratio_)
# Calculate cumulative explained variance ratio
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
# Get the loadings of each feature on the first principal component
loadings = pca.components_[0]

# Identify the most important features by the absolute value of the loadings
important_features = X.columns[np.abs(loadings).argsort()[::-1]]
print(important_features)
# Plot the cumulative explained variance ratio to determine number of components
# plt.plot(cumulative_var)
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.show()
# # Plot the first two principal components
plt.scatter(data_transformed[:, 0], data_transformed[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

#EDA

# combinations['geo_Germany_x'].value_counts().plot(kind='bar')
# plt.title('Location Frequency Distribution')
# plt.xlabel('geo_Germany_x')
# plt.ylabel('Frequency')
# plt.show()

# # Calculate the correlation between Age and Match
# age_match_corr = combinations['geo_Germany_x'].corr(combinations['services_Marketing Solutions_x'])
# print('Correlation between Germany_x and Marketing solution:', age_match_corr)
# correlation graph with survival
correlation = combinations.iloc[:, 1:].corr()['Match'].sort_values(ascending=False)

correlation[1:].plot(kind='bar', figsize=(12,6), title='Correlation with Match')
plt.show()