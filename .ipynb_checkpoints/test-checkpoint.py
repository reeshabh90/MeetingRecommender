import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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
# Merge the matched profiles
merged = matches.merge(profiles_encoded.add_suffix('_P1'),
                        left_on='P1 Confirmation Number',
                        right_on='Confirmation Number_P1')
merged = merged.merge(profiles_encoded.add_suffix('_P2'),
                        left_on='P2 Confirmation Number',
                        right_on='Confirmation Number_P2')
merged = merged.drop(['P1 Name', 'P1 Confirmation Number', 'P2 Name', 'P2 Confirmation Number', 'Meeting Number'], axis=1)
merged.to_excel("merged2.xlsx", index=False)
geo_cols = [col for col in merged.columns if col.startswith('geo_')]
req_cols = [col for col in merged.columns if col.startswith('services_req_')]
off_cols = [col for col in merged.columns if col.startswith('services_')]
# # Define the input features and target variable
X = merged[geo_cols]
y = merged['Match'].fillna(0)

# Train a random forest classification model
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

print(feature_importances)
#print(top_10_features)
corr = merged[geo_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
