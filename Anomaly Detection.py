import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt


# Load data
Pre_data = pd.read_csv("")

# Selecting columns
columns_to_exclude = ['PI']
selected_columns = [col for col in Pre_data.columns if col not in columns_to_exclude]

data = Pre_data[selected_columns]
data = data.fillna(data.mean())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Model Training
model_isolation_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42)
model_isolation_forest.fit(scaled_data)

model_one_class_svm = OneClassSVM(nu=0.1, kernel="rbf",gamma=0.1)
model_one_class_svm.fit(scaled_data)

model_elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
model_elliptic_envelope.fit(scaled_data)

model_SGDone_class_svm =linear_model.SGDOneClassSVM(random_state=42)
model_SGDone_class_svm.fit(scaled_data)

# Predictions
data['scores_isolation_forest'] = model_isolation_forest.decision_function(scaled_data)
data['anomaly_isolation_forest'] = model_isolation_forest.predict(scaled_data)
data['scores_one_class_svm'] = model_one_class_svm.decision_function(scaled_data)
data['anomaly_one_class_svm'] = model_one_class_svm.predict(scaled_data)
data['scores_elliptic_envelope'] = model_elliptic_envelope.decision_function(scaled_data)
data['anomaly_elliptic_envelope'] = model_elliptic_envelope.predict(scaled_data)
data['scores_SGDone_class_svm'] = model_SGDone_class_svm.decision_function(scaled_data)
data['anomaly_SGDone_class_svm'] = model_SGDone_class_svm.predict(scaled_data)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display Anomalies
print("Anomalies detected by Isolation Forest:")
print(data[data['anomaly_isolation_forest'] == -1])

print("\nAnomalies detected by One-Class SVM:")
print(data[data['anomaly_one_class_svm'] == -1])

print("\nAnomalies detected by Elliptic Envelope:")
print(data[data['anomaly_elliptic_envelope'] == -1])

print("\nAnomalies detected SGD one class SVM:")
print(data[data['anomaly_SGDone_class_svm'] == -1])
# Visualization
plt.scatter(data.index, data['scores_isolation_forest'])
plt.xlabel('Data Index')
plt.ylabel('Anomaly Score')
plt.title('Isolation Forest Anomaly Scores')
plt.savefig('isolation_forest_scores.png', dpi=600)  # Adjust DPI as needed
plt.show()

plt.scatter(data.index, data['scores_one_class_svm'])
plt.xlabel('Data Index')
plt.ylabel('Anomaly Score')
plt.title('One class SVM Anomaly Scores')
plt.savefig('One_class_svm_scores.png', dpi=600)  # Adjust DPI as needed
plt.show()

plt.scatter(data.index, data['scores_elliptic_envelope'])
plt.xlabel('Data Index')
plt.ylabel('Anomaly Score')
plt.title('Eliptic Envelope Anomaly Scores')
plt.savefig('scores_elliptic_envelope.png', dpi=600)  # Adjust DPI as needed
plt.show()

plt.scatter(data.index, data['scores_one_class_svm'])
plt.xlabel('Data Index')
plt.ylabel('Anomaly Score')
plt.title('One Class SVM Anomaly Scores')
plt.savefig('scores_SGDone_class_svm.png', dpi=600)  # Adjust DPI as needed
plt.show()

data.to_csv("", index=False)

# Anomalies common to all three algorithms
common_anomalies = data[(data['anomaly_isolation_forest'] == -1) &
                        (data['anomaly_one_class_svm'] == -1) &
                        (data['anomaly_elliptic_envelope'] == -1)&
                        (data['anomaly_SGDone_class_svm'] == -1)]

# Export Common Anomalies to CSV
common_anomalies.to_csv("", index=False)

# Print the number of common anomalies found
print("Number of common anomalies found:", len(common_anomalies))


from scipy.stats import ttest_ind, mannwhitneyu
anomaly_column = 'anomaly_SGDone_class_svm'
results_df = pd.DataFrame(columns=['Feature', 'T_Stat', 'P_Value_TTest', 'U_Stat', 'P_Value_MannWhitneyU'])

for feature_to_compare in data.columns.difference([anomaly_column]):
    normal_data = data[data[anomaly_column] == 1][feature_to_compare]
    anomalous_data = data[data[anomaly_column] == -1][feature_to_compare]

    t_stat, p_value_ttest = ttest_ind(normal_data, anomalous_data, equal_var=False)

    u_stat, p_value_mannwhitneyu = mannwhitneyu(normal_data, anomalous_data, alternative='two-sided')

    results_df = results_df._append({
        'Feature': feature_to_compare,
        'T_Stat': t_stat,
        'P_Value_TTest': p_value_ttest,
        'U_Stat': u_stat,
        'P_Value_MannWhitneyU': p_value_mannwhitneyu
    }, ignore_index=True)

results_df.to_csv("", index=False)
