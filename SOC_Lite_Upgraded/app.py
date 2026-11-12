"""SOC Lite ML-IDS: synthetic demo + genuine NSL-KDD/CICIDS2017 dataset ingestion."""
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

from alert_generator import SOCAlertGenerator
from dataset_loader import read_cicids2017, read_nsl_kdd
from model_engine import SOCModelEngine
from preprocessor import NetworkDataPreprocessor, generate_sample_nslkdd_data

st.set_page_config(page_title="SOC Lite ML-IDS", layout="wide")
st.title("🛡️ SOC Lite ML-IDS — Giám sát an ninh mạng bằng Machine Learning")
st.caption("Hỗ trợ bộ dữ liệu NSL-KDD và CICIDS2017 gốc; mô phỏng kiểm thử luồng mạng bằng dữ liệu đã nạp.")

for key, default in {
    "raw_df": None, "official_test_df": None, "dataset_kind": None,
    "dataset_name": "", "preprocessor": None, "engine": None,
    "results": None, "processed_preview": None, "test_raw": None,
    "test_labels": None, "train_count": None, "test_count": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def set_data(train_df, test_df, kind, name):
    st.session_state.raw_df = train_df
    st.session_state.official_test_df = test_df
    st.session_state.dataset_kind = kind
    st.session_state.dataset_name = name
    for key in ("preprocessor", "engine", "results", "processed_preview",
                "test_raw", "test_labels", "train_count", "test_count"):
        st.session_state[key] = None


# ------------------------------- FR-1 -------------------------------
st.sidebar.header("1. Chọn nguồn dữ liệu (FR-1)")
source = st.sidebar.radio("Nguồn dữ liệu", [
    "Mô phỏng (chỉ để demo)",
    "NSL-KDD thật — KDDTrain+/KDDTest+",
    "CICIDS2017 thật — MachineLearningCSV",
])

if source.startswith("Mô phỏng"):
    n_samples = st.sidebar.slider("Số bản ghi mô phỏng", 500, 5000, 1500, step=500)
    if st.sidebar.button("Tạo dữ liệu mô phỏng", type="primary"):
        set_data(generate_sample_nslkdd_data(n_samples), None, "demo", "Dữ liệu mô phỏng (KHÔNG phải NSL-KDD thật)")
        st.sidebar.success("Đã khởi tạo dữ liệu mô phỏng.")

elif source.startswith("NSL-KDD"):
    st.sidebar.info("Cần cả KDDTrain+.txt và KDDTest+.txt. Giữ nguyên tập Test chuẩn để đánh giá.")
    source_method = st.sidebar.radio("Cách đọc NSL-KDD", ["Tải lên file", "Đọc file từ thư mục máy"], key="nsl_method")
    if source_method == "Tải lên file":
        train_file = st.sidebar.file_uploader("KDDTrain+.txt", type=["txt", "csv"], key="nsl_train")
        test_file = st.sidebar.file_uploader("KDDTest+.txt", type=["txt", "csv"], key="nsl_test")
    else:
        train_file = st.sidebar.text_input("Đường dẫn tập Train", "data/nsl_kdd/KDDTrain+.txt")
        test_file = st.sidebar.text_input("Đường dẫn tập Test", "data/nsl_kdd/KDDTest+.txt")
    if st.sidebar.button("Nạp NSL-KDD thật", type="primary"):
        try:
            if not train_file or not test_file:
                raise ValueError("Hãy chọn đầy đủ Train và Test.")
            train_df = read_nsl_kdd(train_file)
            test_df = read_nsl_kdd(test_file)
            if len(train_df) < 2 or len(test_df) < 1:
                raise ValueError("Bộ dữ liệu không đủ bản ghi.")
            set_data(train_df, test_df, "nsl_kdd", "NSL-KDD thật (Train/Test chuẩn)")
            st.sidebar.success(f"Nạp xong: Train {len(train_df):,} / Test {len(test_df):,} bản ghi")
        except Exception as exc:
            st.sidebar.error(f"Không nạp được NSL-KDD: {exc}")

else:
    st.sidebar.info("Chọn CSV trong gói MachineLearningCSV (không phải PCAP). Nên nạp nhiều ngày để có cả Normal và Attack.")
    source_method = st.sidebar.radio("Cách đọc CICIDS2017", ["Tải lên file", "Đọc file từ thư mục máy"], key="cic_method")
    max_rows_per_file = st.sidebar.select_slider(
        "Số dòng lấy mẫu tối đa / file", options=[2000, 5000, 10000, 20000, 50000, 100000], value=20000
    )
    if source_method == "Tải lên file":
        cic_sources = st.sidebar.file_uploader("Chọn một hoặc nhiều file CICIDS2017 .csv", type="csv", accept_multiple_files=True)
    else:
        folder = st.sidebar.text_input("Thư mục CSV trên máy", "data/cicids2017")
        cic_sources = sorted(Path(folder).glob("*.csv")) if folder else []
        st.sidebar.caption(f"Tìm thấy {len(cic_sources)} file CSV trong thư mục.")
    if st.sidebar.button("Nạp CICIDS2017 thật", type="primary"):
        try:
            data = read_cicids2017(cic_sources, max_rows_per_file=max_rows_per_file)
            set_data(data, None, "cicids2017", f"CICIDS2017 thật ({len(cic_sources)} file CSV, lấy mẫu tối đa {max_rows_per_file:,}/file)")
            st.sidebar.success(f"Nạp xong {len(data):,} bản ghi.")
        except Exception as exc:
            st.sidebar.error(f"Không nạp được CICIDS2017: {exc}")

if st.session_state.raw_df is not None:
    st.sidebar.divider()
    st.sidebar.caption("Dữ liệu đang dùng:")
    st.sidebar.write(st.session_state.dataset_name)
    st.sidebar.caption("Nếu thay nguồn, nhấn nút Nạp/Tạo để cập nhật dữ liệu.")

# ------------------------------- TABS -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Tab 1: Nạp & Tiền xử lý (FR-1, FR-2)",
    "🤖 Tab 2: Huấn luyện & Đánh giá (FR-3, FR-4)",
    "🚨 Tab 3: Dự đoán & Cảnh báo SOC (FR-5)",
])

with tab1:
    if st.session_state.raw_df is None:
        st.info("Chọn nguồn dữ liệu và nhấn nút Nạp/Tạo dữ liệu ở thanh bên trái để bắt đầu.")
    else:
        df = st.session_state.raw_df
        st.subheader(f"Nguồn: {st.session_state.dataset_name}")
        st.caption(f"{len(df):,} mẫu huấn luyện | {len(df.columns) - 1} đặc trưng gốc"
                   + (f" | {len(st.session_state.official_test_df):,} mẫu Test chuẩn"
                      if st.session_state.official_test_df is not None else ""))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dữ liệu ban đầu (8 dòng đầu)**")
            st.dataframe(df.head(8), use_container_width=True)
        with c2:
            st.markdown("**Phân bố nhãn huấn luyện**")
            st.bar_chart(df["label"].value_counts())
        missing_count = int(df.drop(columns="label").isna().sum().sum())
        st.caption(f"Giá trị thiếu trong tập Train trước tiền xử lý: {missing_count:,} ô; "
                   "giá trị vô cực sẽ được xử lý trong pipeline.")
        st.divider()
        st.markdown("**Quy trình chống rò rỉ dữ liệu (NFR-1)**")
        st.write("1. Tách Train/Test trước (hoặc giữ Train/Test gốc của NSL-KDD). "
                 "2. Fit Imputer, One-Hot, Scaler và Feature Selector CHỈ trên Train. "
                 "3. Chỉ Transform trên Test và các bản ghi dự đoán mới.")
        if st.session_state.processed_preview is None:
            st.info("Bảng dữ liệu đã xử lý sẽ xuất hiện sau khi huấn luyện ở Tab 2; không fit trên cả bộ dữ liệu tại Tab 1.")
        else:
            prep = st.session_state.preprocessor
            st.success(f"Đã mã hóa {len(prep.encoded_feature_names)} đặc trưng; "
                       f"chọn {len(prep.selected_feature_names)} đặc trưng cho mô hình.")
            st.dataframe(st.session_state.processed_preview, use_container_width=True)
            st.markdown("**Đặc trưng được chọn (Top-K, ANOVA F)**")
            selected_scores = prep.feature_scores_.loc[
                prep.feature_scores_["Đặc trưng"].isin(prep.selected_feature_names)
            ].head(30)
            st.dataframe(selected_scores, use_container_width=True)

with tab2:
    if st.session_state.raw_df is None:
        st.warning("Hãy nạp dữ liệu ở Tab 1 trước khi huấn luyện.")
    else:
        st.subheader("Huấn luyện Decision Tree, Random Forest và Naive Bayes")
        c1, c2 = st.columns(2)
        with c1:
            scaling = st.selectbox("Cách chuẩn hóa dữ liệu số", ["MinMaxScaler", "StandardScaler"])
        with c2:
            use_selection = st.checkbox("Chọn lọc đặc trưng Top-K bằng ANOVA F", value=True)
        top_k = st.slider("Số đặc trưng giữ lại (sau One-Hot)", 5, 120, 25, step=5, disabled=not use_selection)
        if st.session_state.official_test_df is None:
            test_size = st.slider("Tỷ lệ dữ liệu kiểm thử", 0.1, 0.4, 0.2, step=0.05)
        else:
            test_size = None
            st.info("NSL-KDD: sử dụng KDDTrain+ để huấn luyện, KDDTest+ để đánh giá (không trộn rồi chia lại).")

        if st.button("🚀 Bắt đầu huấn luyện 3 mô hình", type="primary"):
            try:
                train_df = st.session_state.raw_df
                if train_df["label"].nunique() < 2:
                    raise ValueError("Tập Train phải có ít nhất hai nhãn khác nhau.")
                if st.session_state.official_test_df is not None:
                    X_train_raw = train_df.drop(columns="label")
                    y_train = train_df["label"]
                    X_test_raw = st.session_state.official_test_df.drop(columns="label")
                    y_test = st.session_state.official_test_df["label"]
                else:
                    X = train_df.drop(columns="label")
                    y = train_df["label"]
                    min_class = int(y.value_counts().min())
                    # In rare/imbalanced samples stratified splitting may be impossible.
                    stratify = y if min_class >= 2 else None
                    if stratify is None:
                        st.warning("Có lớp chỉ 1 mẫu, không thể chia phân tầng. Nên nạp thêm dữ liệu.")
                    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify
                    )
                prep = NetworkDataPreprocessor(
                    scaler_type=scaling, top_k=top_k if use_selection else None
                )
                with st.spinner("Đang fit tiền xử lý trên Train và huấn luyện mô hình..."):
                    X_train = prep.fit_transform(X_train_raw, y_train)
                    X_test = prep.transform(X_test_raw)
                    engine = SOCModelEngine()
                    results = engine.train_and_evaluate(X_train, y_train, X_test, y_test)
                st.session_state.preprocessor = prep
                st.session_state.engine = engine
                st.session_state.results = results
                st.session_state.processed_preview = X_train.head(8)
                st.session_state.test_raw = X_test_raw.reset_index(drop=True)
                st.session_state.test_labels = y_test.reset_index(drop=True)
                st.session_state.train_count = len(X_train)
                st.session_state.test_count = len(X_test)
                st.success("Huấn luyện và đánh giá thành công! Kết quả bên dưới là số liệu tính thực tế từ tập dữ liệu được nạp.")
            except Exception as exc:
                st.error(f"Lỗi huấn luyện / tiền xử lý: {exc}")

        if st.session_state.results is not None:
            st.caption(f"Train: {st.session_state.train_count:,} | Test: {st.session_state.test_count:,} "
                       f"| đặc trưng cuối: {len(st.session_state.preprocessor.selected_feature_names)}")
            res = st.session_state.results
            metrics_df = pd.DataFrame({
                name: {key: val for key, val in data.items()
                       if key not in ("ConfusionMatrix", "Classes")}
                for name, data in res.items()
            }).T
            st.dataframe(metrics_df, use_container_width=True)
            col_cm, col_importance = st.columns(2)
            with col_cm:
                cm_name = st.selectbox("Chọn mô hình xem Confusion Matrix", list(res.keys()), index=1)
                cm = res[cm_name]["ConfusionMatrix"]
                classes = res[cm_name]["Classes"]
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                            yticklabels=classes, ax=ax)
                ax.set_xlabel("Dự đoán")
                ax.set_ylabel("Thực tế")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col_importance:
                st.markdown("**Độ quan trọng đặc trưng (Random Forest)**")
                imp = st.session_state.engine.get_feature_importance("Random Forest")
                if not imp.empty:
                    st.bar_chart(imp.head(25).set_index("Đặc trưng (Feature)"))
                st.caption("Điểm ANOVA F phục vụ chọn lọc đặc trưng hiển thị ở Tab 1.")

with tab3:
    if st.session_state.results is None:
        st.warning("Huấn luyện mô hình ở Tab 2 trước khi dự đoán.")
    else:
        st.subheader("Dự đoán luồng mạng và phát sinh cảnh báo SOC")
        engine = st.session_state.engine
        name = st.selectbox("Mô hình suy luận", list(engine.trained_models.keys()), index=1)
        model = engine.trained_models[name]
        if st.session_state.dataset_kind == "demo":
            st.info("Chế độ mô phỏng: nhập đặc trưng thủ công, không phải gói tin mạng đang được bắt trực tiếp.")
            presets = {
                "Normal HTTP": ("tcp", "http", "SF", 12, 1250, 3400, 8, 8, 0.0, 1.0),
                "DoS": ("tcp", "http", "S0", 0, 0, 0, 380, 350, 0.98, 0.05),
                "Probe": ("tcp", "private", "REJ", 0, 20, 0, 180, 4, 0.45, 0.04),
                "R2L": ("tcp", "ftp", "SF", 120, 950, 2100, 4, 4, 0.05, 0.90),
                "U2R": ("tcp", "telnet", "SF", 280, 2400, 6200, 2, 2, 0.0, 1.0),
            }
            preset = st.selectbox("Kịch bản mẫu", list(presets))
            default = presets[preset]
            c1, c2 = st.columns(2)
            with c1:
                proto = st.selectbox("protocol_type", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(default[0]))
                flag = st.selectbox("flag", ["SF", "S0", "REJ", "RSTR"], index=["SF", "S0", "REJ", "RSTR"].index(default[2]))
                src = st.number_input("src_bytes", 0, 100000, default[4])
                count = st.slider("count", 0, 511, default[6])
                error = st.slider("serror_rate", 0.0, 1.0, default[8])
            with c2:
                service = st.selectbox("service", ["http", "ftp", "smtp", "dns", "private", "telnet", "ssh", "other"], index=["http", "ftp", "smtp", "dns", "private", "telnet", "ssh", "other"].index(default[1]))
                duration = st.number_input("duration", 0, 100000, default[3])
                dst = st.number_input("dst_bytes", 0, 100000, default[5])
                srv_count = st.slider("srv_count", 0, 511, default[7])
                same = st.slider("same_srv_rate", 0.0, 1.0, default[9])
            flow = pd.DataFrame([{
                "duration": duration, "protocol_type": proto, "service": service,
                "flag": flag, "src_bytes": src, "dst_bytes": dst,
                "count": count, "srv_count": srv_count, "serror_rate": error,
                "same_srv_rate": same,
            }])
            actual_label = None
        else:
            st.info("Chế độ dữ liệu thật: chọn một bản ghi từ tập Test đã giữ lại. "
                    "Chưa có chức năng bắt gói tin mạng trực tiếp.")
            idx = st.number_input("Chọn chỉ số bản ghi trong Test (bắt đầu từ 0)",
                                  min_value=0, max_value=len(st.session_state.test_raw) - 1,
                                  value=0, step=1)
            flow = st.session_state.test_raw.iloc[[int(idx)]]
            actual_label = st.session_state.test_labels.iloc[int(idx)]
            with st.expander("Xem đặc trưng thực tế của bản ghi"):
                st.dataframe(flow.T.rename(columns={flow.index[0]: "Giá trị"}), use_container_width=True)
            st.caption(f"Nhãn gốc trong Test: {actual_label}")
            proto = str(flow.iloc[0].get("protocol_type", flow.iloc[0].get("Protocol", "unknown")))
            service = str(flow.iloc[0].get("service", flow.iloc[0].get("Destination Port", "unknown")))
        if st.button("🔎 Phân tích và phát cảnh báo", type="primary"):
            try:
                X_flow = st.session_state.preprocessor.transform(flow)
                pred = str(model.predict(X_flow)[0])
                proba = float(np.max(model.predict_proba(X_flow))) if hasattr(model, "predict_proba") else 0.0
                alert = SOCAlertGenerator().create_alert(
                    f"FLOW_{uuid.uuid4().hex[:12].upper()}", pred, proba, proto, service
                )
                if pred == "Normal":
                    st.success(f"✅ Normal | độ tin cậy {proba:.2%} | {alert['recommended_action']}")
                else:
                    st.error(f"🚨 {pred} | độ tin cậy {proba:.2%} | "
                             f"{alert['severity']} | {alert['recommended_action']}")
                if st.session_state.dataset_kind == "cicids2017":
                    st.caption("CICIDS2017 giữ nhãn tấn công nguyên bản; Severity của các nhãn "
                               "chưa được ánh xạ riêng mặc định là MEDIUM/LOG_ONLY (chỉ để demo).")
                if actual_label is not None:
                    st.write(f"Nhãn thật: **{actual_label}** — Dự đoán: **{pred}**")
                st.json(alert)
            except Exception as exc:
                st.error(f"Không dự đoán được bản ghi: {exc}")
