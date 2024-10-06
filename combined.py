import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore


csv_file = input('Enter csv file path: ')
data_cat = pd.read_csv(csv_file)


csv_times = np.array(data_cat['time_rel(sec)'].tolist())
csv_data = np.array(data_cat['velocity(m/s)'].tolist())


z_scores = zscore(csv_data)


threshold = 5
anomalies = np.abs(z_scores) > threshold


window_size = 1000
num_anomalies_per_window = [
    np.sum(anomalies[i:i + window_size])
    for i in range(0, len(csv_data), window_size)
]


max_anomalies_window_index = np.argmax(num_anomalies_per_window)
start_idx = max_anomalies_window_index * window_size


end_idx = None
for i in range(start_idx, len(z_scores)):
    if np.abs(z_scores[i]) < 0.1:
        end_idx = i
        break


if end_idx is not None:
    anomaly_duration = csv_times[end_idx] - csv_times[start_idx]
    print(f"Duration of the anomaly: {anomaly_duration} seconds")

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(csv_times, csv_data, label='Velocity (m/s)')

ax.axvline(x=csv_times[start_idx], color='red', linestyle='--', label='Start of anomaly')

# Настройка графика
ax.set_xlim([min(csv_times), max(csv_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{csv_file}', fontweight='bold')
ax.legend()

plt.show()


if start_idx + 1 < len(csv_times):
    right_csv_times = csv_times[start_idx + 1:]
    right_csv_data = csv_data[start_idx + 1:]

    fig_right, ax_right = plt.subplots(1, 1, figsize=(10, 3))
    ax_right.plot(right_csv_times, right_csv_data, label='Velocity (m/s)', color='orange')

    ax_right.set_xlim([min(right_csv_times), max(right_csv_times)])
    ax_right.set_ylabel('Velocity (m/s)')
    ax_right.set_xlabel('Time (s)')
    ax_right.set_title('Anomaly overview', fontweight='bold')
    ax_right.legend()

    plt.show()


from sklearn.cluster import KMeans
import os

print('Using K-means to compare this anomaly')
def extract_features(csv_file):
    data = pd.read_csv(csv_file)
    velocity = data['velocity(m/s)'].values


    features = {
        'mean_velocity': np.mean(velocity),
        'max_velocity': np.max(velocity),
        'min_velocity': np.min(velocity),
        'std_velocity': np.std(velocity),
        'duration': len(velocity)
    }
    return features


def load_all_csv_features(directory):
    features_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            features = extract_features(file_path)
            features['file_name'] = filename
            features_list.append(features)
    return pd.DataFrame(features_list)



directory = 'csvfiles'
data_features = load_all_csv_features(directory)


kmeans = KMeans(n_clusters=5)
kmeans.fit(data_features[['mean_velocity', 'std_velocity']])

data_features['cluster'] = kmeans.labels_


def plot_clusters(features_df, new_file=None):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_df['mean_velocity'], features_df['std_velocity'], c=features_df['cluster'],
                          cmap='viridis', label='Existing Files')
    plt.colorbar(scatter, label='Cluster')

    if new_file:
        new_features = extract_features(new_file)
        new_features_array = np.array([[new_features['mean_velocity'], new_features['std_velocity']]])
        cluster = kmeans.predict(new_features_array)

        plt.scatter(new_features['mean_velocity'], new_features['std_velocity'], color='red', marker='x', s=100,
                    label='New File')
        plt.title(f"New file '{new_file}' belongs to cluster: {cluster[0]}")

    plt.xlabel('Mean Velocity')
    plt.ylabel('Standard Deviation of Velocity')
    plt.title('Clustering of Seismic Activities')
    plt.legend()
    plt.show()


new_file = csv_file
plot_clusters(data_features, new_file)

print('Comparison ended!')

