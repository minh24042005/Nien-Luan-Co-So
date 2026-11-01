import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class NetworkDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        self.numeric_cols = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Huấn luyện bộ tiền xử lý và biến đổi tập dữ liệu Train (Chống Data Leakage)"""
        df_clean = df.copy()

        # 1. Xử lý giá trị vô cực (inf) và giá trị khuyết (NaN)
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

        # 2. Mã hóa biến phân loại dạng chuỗi sang số (Label Encoding)
        for col in self.categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        # 3. Chuẩn hóa thang đo đặc trưng số học về [0, 1] bằng MinMaxScaler
        self.numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in self.numeric_cols:
            self.numeric_cols.remove('label')

        df_clean[self.numeric_cols] = self.scaler.fit_transform(df_clean[self.numeric_cols])
        return df_clean

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng bộ thông số đã học sang tập dữ liệu Test hoặc gói tin thời gian thực"""
        df_clean = df.copy()
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.fillna(0, inplace=True)

        for col in self.categorical_cols:
            if col in df_clean.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df_clean[col] = df_clean[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )

        df_clean[self.numeric_cols] = self.scaler.transform(df_clean[self.numeric_cols])
        return df_clean


def generate_sample_nslkdd_data(n_samples: int = 1500) -> pd.DataFrame:
    """Hàm tạo dữ liệu lưu lượng mạng mô phỏng chuẩn cấu trúc NSL-KDD để chạy Demo ngay"""
    np.random.seed(42)
    records = []
    # Tỷ lệ mô phỏng thực tế (có tính chất mất cân bằng lớp Class Imbalance)
    categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    probabilities = [0.55, 0.28, 0.10, 0.05, 0.02]

    for _ in range(n_samples):
        label = np.random.choice(categories, p=probabilities)
        if label == 'Normal':
            proto = np.random.choice(['tcp', 'udp'], p=[0.8, 0.2])
            service = np.random.choice(['http', 'smtp', 'dns', 'ftp'])
            flag = 'SF'
            duration = np.random.randint(0, 50)
            src_bytes = np.random.randint(150, 5000)
            dst_bytes = np.random.randint(200, 8000)
            count = np.random.randint(1, 25)
            srv_count = np.random.randint(1, 25)
            serror_rate = np.round(np.random.uniform(0.0, 0.05), 2)
            same_srv_rate = np.round(np.random.uniform(0.85, 1.0), 2)
        elif label == 'DoS':
            proto = np.random.choice(['tcp', 'icmp'], p=[0.75, 0.25])
            service = np.random.choice(['http', 'private', 'ecr_i'])
            flag = np.random.choice(['S0', 'REJ', 'SF'], p=[0.7, 0.2, 0.1])
            duration = 0
            src_bytes = np.random.choice([0, 1032, 52000])
            dst_bytes = 0
            count = np.random.randint(150, 511)
            srv_count = np.random.randint(100, 511)
            serror_rate = np.round(np.random.uniform(0.80, 1.0), 2)
            same_srv_rate = np.round(np.random.uniform(0.0, 0.3), 2)
        elif label == 'Probe':
            proto = np.random.choice(['tcp', 'udp', 'icmp'])
            service = np.random.choice(['other', 'private', 'http', 'ssh'])
            flag = np.random.choice(['REJ', 'S0', 'RSTR'])
            duration = np.random.randint(0, 5)
            src_bytes = np.random.randint(0, 100)
            dst_bytes = 0
            count = np.random.randint(50, 250)
            srv_count = np.random.randint(1, 15)
            serror_rate = np.round(np.random.uniform(0.2, 0.6), 2)
            same_srv_rate = np.round(np.random.uniform(0.01, 0.25), 2)
        elif label == 'R2L':
            proto = 'tcp'
            service = np.random.choice(['ftp', 'ftp_data', 'telnet', 'ssh'])
            flag = 'SF'
            duration = np.random.randint(10, 300)
            src_bytes = np.random.randint(100, 3000)
            dst_bytes = np.random.randint(100, 5000)
            count = np.random.randint(1, 10)
            srv_count = np.random.randint(1, 10)
            serror_rate = np.round(np.random.uniform(0.0, 0.15), 2)
            same_srv_rate = np.round(np.random.uniform(0.7, 1.0), 2)
        else:  # U2R
            proto = 'tcp'
            service = np.random.choice(['telnet', 'ftp', 'ssh'])
            flag = 'SF'
            duration = np.random.randint(30, 600)
            src_bytes = np.random.randint(500, 4500)
            dst_bytes = np.random.randint(1000, 9000)
            count = np.random.randint(1, 5)
            srv_count = np.random.randint(1, 5)
            serror_rate = 0.0
            same_srv_rate = 1.0

        records.append({
            'duration': duration,
            'protocol_type': proto,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'same_srv_rate': same_srv_rate,
            'label': label
        })

    return pd.DataFrame(records)