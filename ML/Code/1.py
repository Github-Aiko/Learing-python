import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp diabetes.csv
df = pd.read_csv('diabetes.csv')

# Chia dữ liệu thành features và target
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính độ chính xác
print('Accuracy: ', accuracy_score(y_test, y_pred))
