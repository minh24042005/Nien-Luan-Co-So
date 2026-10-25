import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="SOC-Lite | AI NIDS Demo", page_icon="🛡️", layout="wide")
st.title("🛡️ SOC-Lite — Demo phát hiện lưu lượng mạng bất thường")
st.caption("Bản demo học tập: tải CSV, huấn luyện mô hình ML, xem chỉ số và dự đoán mẫu mới.")

with st.sidebar:
    st.header("Cấu hình")
    uploaded = st.file_uploader("Tải bộ dữ liệu CSV", type=["csv"])
    st.markdown("---")
    st.write("**Lưu ý:** Chỉ dùng dữ liệu được phép sử dụng. Không tải PCAP/log chứa thông tin nhạy cảm lên nơi công cộng.")

if uploaded is None:
    st.info("Hãy tải lên file CSV của bộ dữ liệu (ví dụ NSL-KDD, CICIDS2017 hoặc UNSW-NB15 đã chuyển thành CSV).")
    st.markdown("""
    ### Demo này làm được gì?
    1. Đọc và xem nhanh dữ liệu.
    2. Chọn cột nhãn cần dự đoán.
    3. Chia Train/Test trước khi học các bước tiền xử lý để hạn chế rò rỉ dữ liệu.
    4. Điền giá trị thiếu và mã hóa biến phân loại bằng pipeline.
    5. Huấn luyện Random Forest, hiển thị Accuracy, Precision, Recall, F1 và ma trận nhầm lẫn.
    6. Ước lượng độ quan trọng đặc trưng bằng Permutation Importance.

    **Đầu vào mong đợi:** mỗi dòng là một bản ghi/flow mạng; một cột là nhãn bình thường/tấn công.
    """)
    st.stop()

try:
    df = pd.read_csv(uploaded, low_memory=False)
except Exception as e:
    st.error(f"Không đọc được CSV: {e}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("CSV cần có ít nhất 2 cột và có dữ liệu.")
    st.stop()

st.subheader("1. Khám phá dữ liệu")
c1, c2, c3 = st.columns(3)
c1.metric("Số bản ghi", f"{len(df):,}")
c2.metric("Số cột", f"{len(df.columns):,}")
c3.metric("Ô trống", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(20), use_container_width=True)

label_col = st.selectbox("Chọn cột nhãn (target)", options=list(df.columns), index=len(df.columns)-1)
test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
n_estimators = st.slider("Số cây Random Forest", min_value=50, max_value=400, value=150, step=50)
run = st.button("🚀 Huấn luyện và đánh giá", type="primary")

if run:
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[label_col])
    y_raw = work[label_col].astype(str).str.strip()
    X = work.drop(columns=[label_col])

    # Remove columns that are completely empty or constant.
    X = X.dropna(axis=1, how="all")
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        st.error("Không còn cột đặc trưng hợp lệ sau khi làm sạch.")
        st.stop()

    # Binary conversion: common normal labels become 0; other labels become attack=1.
    normal_names = {"normal", "benign", "0", "normal."}
    y_lower = y_raw.str.lower()
    unique_labels = sorted(y_raw.unique().tolist())
    if len(unique_labels) < 2:
        st.error("Cột nhãn chỉ có một lớp; cần ít nhất hai lớp.")
        st.stop()

    if len(unique_labels) == 2:
        # Treat known normal label as 0; otherwise use the first sorted class as 0.
        normal_mask = y_lower.isin(normal_names)
        if normal_mask.any():
            y = (~normal_mask).astype(int)
        else:
            mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = y_raw.map(mapping).astype(int)
            st.warning(f"Không nhận diện được nhãn 'normal/benign'. Ánh xạ đang dùng: {mapping}")
    else:
        # For multiclass source labels, convert to binary normal vs attack.
        normal_mask = y_lower.isin(normal_names)
        if not normal_mask.any():
            st.error("Phát hiện hơn 2 nhãn nhưng không nhận diện được lớp bình thường. "
                     "Hãy đổi nhãn thành 'Normal' và tên các lớp tấn công, hoặc dùng dữ liệu nhị phân.")
            st.stop()
        y = (~normal_mask).astype(int)

    # Convert obvious numeric columns; leave other columns categorical.
    for col in X.columns:
        if X[col].dtype == "object":
            numeric_candidate = pd.to_numeric(X[col], errors="coerce")
            if numeric_candidate.notna().mean() >= 0.9:
                X[col] = numeric_candidate

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Split before fitting imputers/encoders to prevent test information leakage.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        st.warning("Không thể chia stratify; đã chia ngẫu nhiên thông thường.")

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    with st.spinner("Đang huấn luyện mô hình..."):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

    st.subheader("2. Kết quả đánh giá trên Test")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, pred)*100:.2f}%")
    m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0)*100:.2f}%")
    m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0)*100:.2f}%")
    m4.metric("F1-Score", f"{f1_score(y_test, pred, zero_division=0)*100:.2f}%")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Ma trận nhầm lẫn")
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Dự đoán: Bình thường", "Dự đoán: Tấn công"])
        ax.set_yticks([0, 1], labels=["Thực tế: Bình thường", "Thực tế: Tấn công"])
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Ô dưới-phải là số mẫu tấn công được phát hiện đúng; ô dưới-trái là số mẫu tấn công bị bỏ sót.")
    with right:
        st.markdown("#### Báo cáo phân loại")
        report = classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Bình thường (0)", "Tấn công (1)"],
            zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.subheader("3. Đặc trưng quan trọng")
    st.write("Permutation Importance đo mức suy giảm điểm số khi xáo trộn một đặc trưng. Kết quả phụ thuộc vào dữ liệu Test và mô hình.")
    with st.spinner("Đang tính độ quan trọng đặc trưng..."):
        try:
            pi = permutation_importance(
                clf, X_test, y_test, n_repeats=3, random_state=42,
                scoring="f1", n_jobs=-1
            )
            imp = pd.DataFrame({
                "Đặc trưng": X_test.columns,
                "Mức quan trọng (giảm F1)": pi.importances_mean
            }).sort_values("Mức quan trọng (giảm F1)", ascending=False).head(15)
            st.dataframe(imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Không tính được độ quan trọng đặc trưng: {e}")

    st.subheader("4. Dự đoán dữ liệu mới")
    st.write("Tải CSV mới có cùng các cột đặc trưng (không bắt buộc có cột nhãn) để mô hình dự đoán.")
    new_file = st.file_uploader("CSV cần dự đoán", type=["csv"], key="new_data")
    if new_file is not None:
        try:
            new_df = pd.read_csv(new_file, low_memory=False)
            missing = [c for c in X.columns if c not in new_df.columns]
            if missing:
                st.error("Thiếu các cột đặc trưng bắt buộc: " + ", ".join(missing[:20]))
            else:
                new_X = new_df[X.columns].copy().replace([np.inf, -np.inf], np.nan)
                new_pred = clf.predict(new_X)
                out = new_df.copy()
                out["prediction"] = np.where(new_pred == 1, "Tấn công", "Bình thường")
                if hasattr(clf, "predict_proba"):
                    out["attack_probability"] = clf.predict_proba(new_X)[:, 1]
                st.dataframe(out.head(100), use_container_width=True)
                st.download_button(
                    "Tải kết quả dự đoán CSV",
                    data=out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="soc_lite_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Không dự đoán được file mới: {e}")

    st.success("Đã hoàn tất demo. Đây là mô hình thử nghiệm trên dữ liệu CSV, chưa phải hệ thống SOC production.")
    st.warning("Không dùng kết quả demo làm bằng chứng hiệu năng thực tế nếu chưa kiểm tra nguồn dữ liệu, cách chia tập, nhãn, rò rỉ dữ liệu và hiệu năng trên lưu lượng mới.")
