import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("c:/Users/golda/Documents/GitHub/Intrusion-Detection-System-with-Machine-learning/malicious_vs_benign.csv")
features = df.drop(columns=["URL", "Type"], errors="ignore")

for col in features.select_dtypes(include="object").columns:
    features[col] = LabelEncoder().fit_transform(features[col].astype(str))

features = features.fillna(0)
X = StandardScaler().fit_transform(features)
y = df["Type"].astype(int).values

similarity = cosine_similarity(X)
k = 5
edge_index = []
for i in range(similarity.shape[0]):
    top_k = np.argsort(similarity[i])[-k-1:-1]
    for j in top_k:
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

data = Data(x=torch.tensor(X, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long))

train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
data.train_mask = torch.zeros(len(y), dtype=torch.bool)
data.test_mask = torch.zeros(len(y), dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask[test_idx] = True

class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(num_features=data.num_features).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
out = model(data)
pred = out.argmax(dim=1).cpu().numpy()
true = data.y.cpu().numpy()
test_pred = pred[data.test_mask.cpu()]
test_true = true[data.test_mask.cpu()]

print(classification_report(test_true, test_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_true, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (GNN)")
plt.tight_layout()
plt.show()
