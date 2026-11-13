import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Labels file load karte hain
labels = pd.read_csv("data/labeled_anomalies.csv")
print("Total channels:", len(labels))
print(labels.head())  # Pehli 5 rows dekhte hain

# 2. Ek channel (P-1) ka train aur test data load karte hain
chan_id = "P-1"
train = np.load(f"data/train/{chan_id}.npy")
test = np.load(f"data/test/{chan_id}.npy")

print(f"\n{chan_id} Train shape:", train.shape)
print(f"{chan_id} Test shape:", test.shape)

# 3. P-1 mein asli anomaly kab aayi thi, uske start aur end steps nikalte hain
row = labels[labels["chan_id"] == chan_id].iloc[0]
anomalies = ast.literal_eval(row["anomaly_sequences"])
print(f"{chan_id} True Anomaly intervals:", anomalies)

# 4. Graph plot karte hain (Column 0 = asli sensor reading)
plt.figure(figsize=(12, 4))
plt.plot(test[:, 0], label="Sensor Reading (Test)", color="blue", linewidth=1)

# Jahan asli anomaly hai, us hisse ko red color se highlight karte hain
for start, end in anomalies:
    plt.axvspan(start, end, color="red", alpha=0.3, label="True Anomaly")

plt.title(f"NASA SMAP Channel {chan_id} - Test Telemetry")
plt.xlabel("Time Steps")
plt.ylabel("Sensor Value")
plt.legend()
plt.tight_layout()
plt.show()