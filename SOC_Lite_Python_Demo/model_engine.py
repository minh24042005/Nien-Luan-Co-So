import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


class SOCModelEngine:
    def __init__(self):
        self.models = {
            'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=15, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
            'Naive Bayes': GaussianNB()
        }
        self.trained_models = {}
        self.feature_names = []

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> dict:
        """Huấn luyện đa mô hình và trả về bảng kết quả đối chứng"""
        self.feature_names = X_train.columns.tolist()
        results = {}
        n_test = len(X_test)

        for name, model in self.models.items():
            # Huấn luyện mô hình
            model.fit(X_train, y_train)

            # Đo thời gian suy luận trung bình trên tập kiểm thử (ms / flow)
            start_time = time.perf_counter()
            y_pred = model.predict(X_test)
            elapsed_ms = ((time.perf_counter() - start_time) * 1000) / max(n_test, 1)

            # Tính toán các chỉ số đánh giá
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

            results[name] = {
                'Accuracy (%)': round(acc, 2),
                'Precision (%)': round(prec, 2),
                'Recall (%)': round(rec, 2),
                'F1-Score (%)': round(f1, 2),
                'Latency (ms/flow)': round(elapsed_ms, 4),
                'ConfusionMatrix': cm,
                'Classes': model.classes_
            }
            self.trained_models[name] = model

        return results

    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """Trích xuất độ quan trọng của đặc trưng giúp giải thích cảnh báo SOC"""
        model = self.trained_models.get(model_name)
        if model is not None and hasattr(model, 'feature_importances_'):
            df_imp = pd.DataFrame({
                'Đặc trưng (Feature)': self.feature_names,
                'Độ quan trọng (Importance)': model.feature_importances_
            }).sort_values(by='Độ quan trọng (Importance)', ascending=False)
            return df_imp
        return pd.DataFrame()