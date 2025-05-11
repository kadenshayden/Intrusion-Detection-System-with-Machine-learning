import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("c:/Users/golda/Documents/GitHub/Intrusion-Detection-System-with-Machine-learning/malicious_vs_benign.csv")

benign = df[df["Type"] == 0]
malicious = df[df["Type"] == 1]

benign_X = benign.drop(columns=["Type", "URL"], errors="ignore")
malicious_X = malicious.drop(columns=["Type", "URL"], errors="ignore")

for col in benign_X.select_dtypes(include="object").columns:
    benign_X[col] = benign_X[col].astype("category").cat.codes
for col in malicious_X.select_dtypes(include="object").columns:
    malicious_X[col] = malicious_X[col].astype("category").cat.codes

benign_X = benign_X.fillna(0)
malicious_X = malicious_X.fillna(0)
scaler = StandardScaler()
benign_X_scaled = scaler.fit_transform(benign_X)
malicious_X_scaled = scaler.transform(malicious_X)

X_train, X_benign_test = train_test_split(benign_X_scaled, test_size=0.2, random_state=42)

_, X_malicious_test = train_test_split(malicious_X_scaled, test_size=0.2, random_state=42)

X_test = np.vstack([X_benign_test, X_malicious_test])
y_test = np.array([0] * len(X_benign_test) + [1] * len(X_malicious_test))

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=32, 
                shuffle=True, 
                validation_split=0.1, 
                verbose=1)

X_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

train_pred = autoencoder.predict(X_train)
train_mse = np.mean(np.power(X_train - train_pred, 2), axis=1)
threshold = np.percentile(train_mse, 95)
print(f"Reconstruction error threshold: {threshold:.6f}")

y_pred = (mse > threshold).astype(int)

print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (AutoEncoder)")
plt.tight_layout()
plt.show()