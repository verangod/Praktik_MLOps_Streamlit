import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Sedang melatih model AI...")

# Data latih: [IPK, Total SKS]
# Ditambah data agar model lebih stabil dan akurat
X_train = np.array([
    # IPK tinggi dan SKS tinggi
    [3.9, 150], [3.8, 148], [3.7, 145], [3.6, 142], [3.5, 144],
    [3.9, 146], [3.8, 150], [3.7, 140], [3.6, 138], [3.5, 140],
    [3.4, 145], [3.9, 155], [3.8, 152], [3.7, 148], [3.6, 150],
    [3.5, 138], [3.4, 140], [3.3, 145], [3.9, 160], [3.8, 155],

    # IPK menengah dan SKS cukup
    [3.4, 138], [3.3, 136], [3.2, 140], [3.1, 132], [3.0, 128],
    [3.3, 140], [3.2, 135], [3.1, 138], [3.0, 135], [3.4, 142],
    [3.2, 130], [3.1, 125], [3.0, 130], [3.3, 132], [3.4, 135],
    [3.2, 145], [3.1, 140], [3.0, 138], [3.3, 128], [3.4, 130],

    # IPK menengah ke bawah
    [2.9, 120], [2.8, 118], [2.7, 114], [2.6, 110], [2.5, 108],
    [2.9, 125], [2.8, 120], [2.7, 118], [2.6, 115], [2.5, 112],
    [2.4, 110], [2.3, 105], [2.2, 100], [2.1, 98], [2.0, 100],
    [2.9, 115], [2.8, 112], [2.7, 110], [2.6, 108], [2.5, 105],

    # IPK rendah dan SKS rendah
    [1.9, 90], [1.8, 88], [1.7, 85], [1.6, 80], [1.5, 75],
    [1.9, 95], [1.8, 92], [1.7, 88], [1.6, 85], [1.5, 80],
    [1.4, 75], [1.3, 70], [1.2, 65], [1.1, 60], [1.0, 55],

    # Variasi data tambahan
    [2.4, 102], [2.8, 116], [3.1, 136], [3.5, 130], [3.7, 135],
    [2.6, 125], [2.9, 110], [3.2, 120], [3.4, 125], [3.6, 130],
    [2.1, 110], [2.3, 115], [2.5, 120], [2.7, 125], [3.0, 145],
    [3.8, 140], [3.9, 135], [3.3, 150], [2.0, 85], [1.8, 100]
])

# Label: 1 = Lulus Tepat Waktu, 0 = Tidak Lulus Tepat Waktu
y_train = np.array([
    # 20 data IPK tinggi
    *([1] * 20),

    # 20 data IPK menengah
    *([1] * 20),

    # 20 data IPK menengah ke bawah
    *([0] * 20),

    # 15 data IPK rendah
    *([0] * 15),

    # 25 data variasi
    0, 1, 1, 1, 1,
    0, 0, 1, 1, 1,
    0, 0, 0, 1, 1,
    1, 1, 1, 0, 0
])

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

print("Akurasi Training:", accuracy_score(y_train, y_pred))
print("\nClassification Report:")
print(classification_report(y_train, y_pred))

# Simpan model (Deployment)
joblib.dump(model, 'model_kelulusan.pkl')
print(f"Sukses! Model disimpan sebagai 'model_kelulusan.pkl' dengan {len(X_train)} data latih.")