import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from preprocessor import NetworkDataPreprocessor, generate_sample_nslkdd_data
from model_engine import SOCModelEngine
from alert_generator import SOCAlertGenerator

st.set_page_config(page_title="SOC Lite ML-IDS Demo", layout="wide")

st.title("🛡️ Hệ thống Giám sát An ninh mạng Ứng dụng Học máy (SOC Lite ML-IDS)")
st.markdown(
    "**Đề tài Niên luận cơ sở:** *Tìm hiểu và nghiên cứu việc ứng dụng AI/Machine Learning "
    "trong hệ thống giám sát an ninh mạng*"
)

# Khởi tạo trạng thái lưu trữ (Session State)
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = NetworkDataPreprocessor()
if "engine" not in st.session_state:
    st.session_state.engine = SOCModelEngine()
if "results" not in st.session_state:
    st.session_state.results = None

# Thanh bên (Sidebar): Nạp dữ liệu (FR-1)
st.sidebar.header("1. Nguồn Dữ liệu Lưu lượng (FR-1)")
data_source = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mô phỏng NSL-KDD (Chạy ngay)", "Tải lên tệp CSV (NSL-KDD / CICIDS2017)"]
)

if data_source == "Dữ liệu mô phỏng NSL-KDD (Chạy ngay)":
    n_samples = st.sidebar.slider("Số lượng bản ghi luồng mạng:", 500, 5000, 1500, step=500)
    if st.sidebar.button("🔄 Khởi tạo & Tiền xử lý Dữ liệu"):
        st.session_state.raw_df = generate_sample_nslkdd_data(n_samples)
        st.session_state.results = None
        st.sidebar.success(f"Đã tạo {n_samples} bản ghi lưu lượng mạng!")
else:
    uploaded_file = st.sidebar.file_uploader("Chọn file .csv", type=["csv", "txt"])
    if uploaded_file is not None:
        st.session_state.raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("Đã tải tệp dữ liệu thành công!")

# Giao diện chính chia làm 3 Tab tương ứng với Pipeline trong Báo cáo
tab1, tab2, tab3 = st.tabs([
    "📊 Tab 1: Tiền xử lý & Chuẩn hóa (FR-1, FR-2)",
    "🤖 Tab 2: Huấn luyện & Đối sánh Mô hình (FR-3, FR-4)",
    "🚨 Tab 3: Giám sát SOC & Cảnh báo Thời gian thực (FR-5)"
])

# ================= TAB 1: TIỀN XỬ LÝ DỮ LIỆU =================
with tab1:
    if st.session_state.raw_df is None:
        st.info("👈 Vui lòng nhấn nút **'Khởi tạo & Tiền xử lý Dữ liệu'** ở thanh bên trái để bắt đầu.")
    else:
        df = st.session_state.raw_df
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dữ liệu lưu lượng thô (Raw Network Flows)")
            st.dataframe(df.head(8), use_container_width=True)
            st.caption(f"Tổng số mẫu: {len(df)} | Số thuộc tính: {len(df.columns) - 1}")

        with col2:
            st.subheader("Phân bố các nhãn lưu lượng (Class Distribution)")
            label_counts = df['label'].value_counts()
            st.bar_chart(label_counts)

        st.divider()
        st.subheader("Dữ liệu sau khi Mã hóa (LabelEncoder) & Chuẩn hóa (MinMaxScaler [0, 1])")
        df_processed = st.session_state.preprocessor.fit_transform(df)
        st.dataframe(df_processed.head(8), use_container_width=True)

# ================= TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ =================
with tab2:
    if st.session_state.raw_df is None:
        st.warning("Vui lòng nạp dữ liệu ở Tab 1 trước khi huấn luyện mô hình.")
    else:
        test_size = st.slider("Tỷ lệ tập Kiểm thử (Test Split):", 0.1, 0.4, 0.2, step=0.05)
        if st.button("🚀 Bắt đầu Huấn luyện Đa Mô hình (Decision Tree, Random Forest, Naive Bayes)"):
            df = st.session_state.raw_df
            X = df.drop(columns=['label'])
            y = df['label']

            # Chia tập Train/Test trước, sau đó fit_transform trên Train và transform trên Test
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            prep = NetworkDataPreprocessor()
            X_train = prep.fit_transform(X_train_raw)
            X_test = prep.transform(X_test_raw)

            st.session_state.preprocessor = prep
            st.session_state.results = st.session_state.engine.train_and_evaluate(
                X_train, y_train, X_test, y_test
            )
            st.success("Hoàn tất huấn luyện và đánh giá đối chứng 3 thuật toán!")

        if st.session_state.results is not None:
            res = st.session_state.results
            st.subheader("Bảng 4.1. Ma trận đối chứng hiệu năng các thuật toán Học máy")
            metrics_df = pd.DataFrame({
                model_name: {
                    k: v for k, v in data.items() if k not in ['ConfusionMatrix', 'Classes']
                }
                for model_name, data in res.items()
            }).T
            st.dataframe(metrics_df, use_container_width=True)

            col_cm, col_fi = st.columns(2)
            with col_cm:
                st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
                selected_cm_model = st.selectbox("Chọn mô hình xem Confusion Matrix:", list(res.keys()), index=1)
                cm = res[selected_cm_model]['ConfusionMatrix']
                classes = res[selected_cm_model]['Classes']
                fig, ax = plt.subplots(figsize=(5, 3.8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
                ax.set_xlabel("Nhãn Dự đoán (Predicted)")
                ax.set_ylabel("Nhãn Thực tế (Actual)")
                st.pyplot(fig)

            with col_fi:
                st.subheader("Độ quan trọng đặc trưng (Feature Importance)")
                fi_df = st.session_state.engine.get_feature_importance('Random Forest')
                if not fi_df.empty:
                    st.bar_chart(fi_df.set_index('Đặc trưng (Feature)'))

# ================= TAB 3: GIÁM SÁT SOC THỜI GIAN THỰC =================
with tab3:
    if st.session_state.results is None:
        st.warning("Vui lòng nhấn nút **Huấn luyện Đa Mô hình** ở Tab 2 trước khi chạy giám sát SOC.")
    else:
        st.subheader("Trình mô phỏng Phân tích Gói tin & Phát sinh Cảnh báo SOC (FR-5)")
        col_input, col_alert = st.columns([1, 1])

        with col_input:
            st.markdown("##### Tùy chỉnh thông số Luồng mạng kiểm thử (Network Flow)")
            active_model_name = st.selectbox(
                "Mô hình AI làm Engine suy luận:",
                list(st.session_state.engine.trained_models.keys()),
                index=1
            )
            preset = st.selectbox(
                "Chọn nhanh kịch bản lưu lượng mẫu:",
                [
                    "Lưu lượng Web hợp lệ (Normal HTTP)",
                    "Tấn công Từ chối dịch vụ (DoS - SYN Flood / Neptune)",
                    "Tấn công Dò quét cổng (Probe - PortScan / Satan)",
                    "Tấn công Dò mật khẩu từ xa (R2L - Brute Force FTP)",
                    "Tấn công Leo thang đặc quyền (U2R - Buffer Overflow)"
                ]
            )

            # Gán giá trị mặc định theo kịch bản để dễ kiểm thử
            defaults = {
                "Lưu lượng Web hợp lệ (Normal HTTP)": ("tcp", "http", "SF", 12, 1250, 3400, 8, 8, 0.0, 1.0),
                "Tấn công Từ chối dịch vụ (DoS - SYN Flood / Neptune)": ("tcp", "http", "S0", 0, 0, 0, 380, 350, 0.98, 0.05),
                "Tấn công Dò quét cổng (Probe - PortScan / Satan)": ("tcp", "private", "REJ", 0, 20, 0, 180, 4, 0.45, 0.04),
                "Tấn công Dò mật khẩu từ xa (R2L - Brute Force FTP)": ("tcp", "ftp", "SF", 120, 950, 2100, 4, 4, 0.05, 0.90),
                "Tấn công Leo thang đặc quyền (U2R - Buffer Overflow)": ("tcp", "telnet", "SF", 280, 2400, 6200, 2, 2, 0.0, 1.0),
            }[preset]

            c1, c2 = st.columns(2)
            proto = c1.selectbox("Giao thức (protocol_type):", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(defaults[0]))
            service = c2.selectbox("Dịch vụ đích (service):", ["http", "ftp", "smtp", "dns", "private", "telnet", "ssh", "other"], index=["http", "ftp", "smtp", "dns", "private", "telnet", "ssh", "other"].index(defaults[1]))
            flag = c1.selectbox("Cờ TCP (flag):", ["SF", "S0", "REJ", "RSTR"], index=["SF", "S0", "REJ", "RSTR"].index(defaults[2]))
            duration = c2.number_input("Thời lượng (duration):", 0, 1000, defaults[3])
            src_bytes = c1.number_input("Số byte gửi (src_bytes):", 0, 100000, defaults[4])
            dst_bytes = c2.number_input("Số byte nhận (dst_bytes):", 0, 100000, defaults[5])
            count = c1.slider("Số kết nối cùng đích (count):", 1, 511, defaults[6])
            srv_count = c2.slider("Số kết nối cùng dịch vụ (srv_count):", 1, 511, defaults[7])
            serror_rate = c1.slider("Tỷ lệ lỗi SYN (serror_rate):", 0.0, 1.0, defaults[8])
            same_srv_rate = c2.slider("Tỷ lệ cùng dịch vụ (same_srv_rate):", 0.0, 1.0, defaults[9])

        with col_alert:
            st.markdown("##### Kết quả Phân tích & Bản ghi Cảnh báo SOC (Bảng 2.1)")
            single_flow_df = pd.DataFrame([{
                'duration': duration, 'protocol_type': proto, 'service': service,
                'flag': flag, 'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
                'count': count, 'srv_count': srv_count,
                'serror_rate': serror_rate, 'same_srv_rate': same_srv_rate
            }])

            # Tiền xử lý gói tin và dự đoán bằng mô hình đã chọn
            X_flow = st.session_state.preprocessor.transform(single_flow_df)
            model = st.session_state.engine.trained_models[active_model_name]
            pred_label = model.predict(X_flow)[0]
            proba = np.max(model.predict_proba(X_flow)) if hasattr(model, "predict_proba") else 0.99

            alert_gen = SOCAlertGenerator()
            flow_id = f"FLOW_2026_{np.random.randint(10000, 99999)}"
            alert_record = alert_gen.create_alert(flow_id, pred_label, proba, proto, service)

            if alert_record["prediction_label"] == "Normal":
                st.success(
                    f"✅ **TRẠNG THÁI AN TOÀN (NORMAL)** | Mức độ: `{alert_record['severity']}` "
                    f"| Khuyến nghị: `{alert_record['recommended_action']}`"
                )
            else:
                st.error(
                    f"🚨 **CẢNH BÁO TẤN CÔNG ({alert_record['attack_category']})** | Mức độ: `{alert_record['severity']}` "
                    f"| Khuyến nghị SOC: `{alert_record['recommended_action']}`"
                )

            st.markdown("**Bản ghi JSON chuẩn hóa xuất ra hệ thống SOC (Đúng quy chuẩn Bảng 2.1):**")
            st.json(alert_record)