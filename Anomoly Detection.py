import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("C:/Users/ethan/Downloads/Hydrolase FINAL 20 jan.csv")



# Preprocessing: Handle missing values and standardize data
data = data.fillna(data.mean())  # Replace missing values with mean
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Training the Isolation Forest model
model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',random_state=42)
model.fit(scaled_data)

# Predictions
data['scores'] = model.decision_function(scaled_data)
data['anomaly'] = model.predict(scaled_data)
anomalies = data[data['anomaly'] == -1]

# I have added this bit so that you can see the whole datasheet
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Display anomalies
print("Anomalies detected:")
print(anomalies)
anomalies.to_csv('Detected anomolies.csv')

# # Visualization (Optional)
plt.scatter(data.index, data['scores'])
plt.xlabel('Data Index')
plt.ylabel('Anomaly Score')

# Adding 'PI' values as labels to each data point
for i, pi_value in enumerate(data['PI']):
    plt.text(data.index[i], data['scores'][i], str(pi_value), fontsize=4, ha='right', va='bottom')

plt.savefig('scatter_plot_final1.png', dpi=600)
plt.show()



# Assuming the script from before has been run up to this point

# Basic statistical analysis
print("\nBasic Statistics of Anomalies:")
print(anomalies.describe())
df2=anomalies.describe()
df2.to_csv('Anomolies decribes.csv')

print("\nBasic Statistics of Normal Data:")
normal_data = data[data['anomaly'] == 1]
print(normal_data.describe())
normal_data.to_csv('Basic data.csv')

# Visualization
# Comparing distributions of a specific feature for anomalies vs normal data
feature_to_compare = "HB"  # Add features according to you

plt.figure(figsize=(12, 6))
sns.displot(normal_data[feature_to_compare], bins=20, label='Normal', color='blue')
sns.displot(anomalies[feature_to_compare], bins=20, label='Anomaly', color='red')
plt.xlabel(feature_to_compare)
plt.ylabel('Density')
plt.title('Comparison of Distributions for ' + feature_to_compare)
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(36, 30))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm',linewidths=.5,annot_kws={"size": 6})
plt.title('Correlation Heatmap of All Features')
plt.savefig("Abnormalities Heatmap.png",dpi=600)
plt.show()

