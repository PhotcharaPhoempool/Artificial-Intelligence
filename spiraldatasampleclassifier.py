import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib.colors import ListedColormap # <--- [เพิ่ม] ต้อง import อันนี้มาสร้างสีเอง

# ==========================================
# 1) Function Generate Dataset (คงเดิม)
# ==========================================
def generate_spiral_data(n_points, n_turns, noise=0.2):
    X = []
    y = []
    for class_idx in range(2):
        i = np.arange(n_points)
        theta = (i / n_points) * (n_turns * 2 * np.pi) + (class_idx * np.pi)
        r = (i / n_points) * n_turns 
        r = r + np.random.normal(0, noise, n_points)
        x_val = r * np.cos(theta)
        y_val = r * np.sin(theta)
        data = np.column_stack((x_val, y_val))
        labels = class_idx * np.ones(n_points)
        X.append(data)
        y.append(labels)
    return np.concatenate(X), np.concatenate(y)

# ==========================================
# Feature Engineering (คงเดิม)
# ==========================================
def add_features(X):
    x = X[:, 0]
    y = X[:, 1]
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    sin_r = np.sin(2 * np.pi * radius)
    cos_r = np.cos(2 * np.pi * radius)
    return np.column_stack((np.sin(angle), np.cos(angle), sin_r, cos_r))

# ==========================================
# 2) เตรียมข้อมูล (คงเดิม)
# ==========================================
np.random.seed(42)
X_train_raw, y_train = generate_spiral_data(n_points=1000, n_turns=2, noise=0.1)
X_test_raw, y_test = generate_spiral_data(n_points=1000, n_turns=4, noise=0.1)
X_train = add_features(X_train_raw)
X_test = add_features(X_test_raw)

# ==========================================
# 3) ออกแบบ Model (คงเดิม)
# ==========================================
model = Sequential([
    Dense(64, input_dim=4, activation='relu'), 
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy', metrics=['accuracy'])

print("Start Training...")
model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0)
print("Training Completed.")

# ==========================================
# 5) ทดสอบ (Inference) (คงเดิม)
# ==========================================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
train_error = 1.0 - train_acc
test_error = 1.0 - test_acc

print(f"\nEvaluation Results:")
print(f"Training Accuracy: {train_acc*100:.2f}% | Error: {train_error:.4f}")
print(f"Testing Accuracy:  {test_acc*100:.2f}% | Error: {test_error:.4f}")

# ==========================================
# 6) Plot ผลลัพธ์ (แก้ไขเรื่องสีตรงนี้!)
# ==========================================
def plot_decision_boundary(X_raw, y, model, title, ax):
    x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points_raw = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_features = add_features(mesh_points_raw) 
    
    Z = model.predict(mesh_points_features, verbose=0)
    Z = Z.reshape(xx.shape)

    # --- [ส่วนที่แก้ไขสี] ---
    # 1. กำหนดสีที่ต้องการ (ฟ้า, ส้ม)
    # ใช้ชื่อสีมาตรฐาน หรือ รหัส Hex ก็ได้ เช่น ['#1f77b4', '#ff7f0e']
    my_colors = ['deepskyblue', 'darkorange'] 
    
    # 2. สร้าง Colormap จากสีที่เราเลือก
    my_cmap = ListedColormap(my_colors)

    # 3. Plot พื้นหลัง (Contour) โดยใช้ Colormap ของเรา
    # alpha=0.4 ทำให้สีพื้นหลังจางลงหน่อย จุดจะได้เด่น
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=my_cmap, alpha=0.4)
    
    # 4. Plot จุด (Scatter) แยกตาม Class เพื่อให้สีตรงเป๊ะ
    # แยกข้อมูล Class 0 และ Class 1
    X_class0 = X_raw[y == 0]
    X_class1 = X_raw[y == 1]

    # พล็อต Class 0 ด้วยสีแรก (ฟ้า)
    ax.scatter(X_class0[:, 0], X_class0[:, 1], c=my_colors[0], 
               edgecolors='k', s=30, linewidth=0.8, label='Class 0')
    
    # พล็อต Class 1 ด้วยสีที่สอง (ส้ม)
    ax.scatter(X_class1[:, 0], X_class1[:, 1], c=my_colors[1], 
               edgecolors='k', s=30, linewidth=0.8, label='Class 1')
    
    # เพิ่ม Legend เพื่อบอกว่าสีไหนคือคลาสไหน
    ax.legend()
    # -----------------------
    
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_decision_boundary(X_train_raw, y_train, model, 
                       f"Figure 1: Training Data (2 Turns)\nAcc: {train_acc*100:.1f}% | Error: {train_error:.4f}", 
                       axes[0])

plot_decision_boundary(X_test_raw, y_test, model, 
                       f"Figure 2: Testing Data (4 Turns)\nAcc: {test_acc*100:.1f}% | Error: {test_error:.4f}", 
                       axes[1])

plt.tight_layout()
plt.show()