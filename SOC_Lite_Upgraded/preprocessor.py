"""Train-only fitted preprocessing with safe categorical handling and feature selection."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class NetworkDataPreprocessor:
    """Impute + One-Hot + Scale + optional Top-K, fitted ONLY on train samples.

    The same fitted values (including imputation medians and category vocabularies)
    are applied to the test set and later inference records.
    """

    def __init__(self, scaler_type: str = "MinMaxScaler", top_k: int | None = 25):
        if scaler_type not in {"MinMaxScaler", "StandardScaler"}:
            raise ValueError("scaler_type phải là MinMaxScaler hoặc StandardScaler")
        self.scaler_type = scaler_type
        self.top_k = top_k
        self.transformer = None
        self.selector = None
        self.input_columns = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.encoded_feature_names = []
        self.selected_feature_names = []
        self.feature_scores_ = pd.DataFrame()

    def _check_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        if not self.input_columns:
            return X.copy()
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Dữ liệu dự đoán thiếu đặc trưng: {missing[:10]}")
        return X.loc[:, self.input_columns].copy()

    @staticmethod
    def _replace_inf(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            else:
                # Explicitly preserve missing strings as NaN, not "nan".
                X[col] = X[col].replace(["Infinity", "-Infinity", "inf", "-inf"], np.nan)
                # Normalize pd.NA and mixed categorical scalars before sklearn.
                X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        return X

    def fit_transform(self, X_train: pd.DataFrame, y_train) -> pd.DataFrame:
        if y_train is None:
            raise ValueError("Phải cung cấp nhãn y_train để chọn đặc trưng trên Train.")
        X = self._replace_inf(self._check_features(X_train))
        self.input_columns = X.columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        if not self.input_columns:
            raise ValueError("Không có cột đặc trưng để huấn luyện.")

        pipelines = []
        if self.numeric_cols:
            pipelines.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", MinMaxScaler() if self.scaler_type == "MinMaxScaler"
                 else StandardScaler()),
            ]), self.numeric_cols))
        if self.categorical_cols:
            pipelines.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.categorical_cols))
        self.transformer = ColumnTransformer(pipelines, remainder="drop", sparse_threshold=0)
        encoded = self.transformer.fit_transform(X)
        self.encoded_feature_names = self.transformer.get_feature_names_out().tolist()
        k = "all" if self.top_k is None else min(int(self.top_k), encoded.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            selected = self.selector.fit_transform(encoded, y_train)
        scores = np.nan_to_num(self.selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_scores_ = (pd.DataFrame({"Đặc trưng": self.encoded_feature_names,
                                              "Điểm ANOVA F": scores})
                                .sort_values("Điểm ANOVA F", ascending=False))
        self.selected_feature_names = [name for name, flag in zip(
            self.encoded_feature_names, self.selector.get_support()) if flag]
        return pd.DataFrame(selected, index=X.index, columns=self.selected_feature_names)

    def transform(self, X_new: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None or self.selector is None:
            raise RuntimeError("Phải gọi fit_transform trên Train trước khi transform.")
        X = self._replace_inf(self._check_features(X_new))
        encoded = self.transformer.transform(X)
        selected = self.selector.transform(encoded)
        return pd.DataFrame(selected, index=X.index, columns=self.selected_feature_names)


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