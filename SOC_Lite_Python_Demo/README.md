# SOC-Lite — Demo AI/ML phát hiện lưu lượng mạng

## Yêu cầu
- Python 3.10 trở lên
- File dữ liệu CSV hợp pháp, mỗi dòng là một flow/bản ghi
- Một cột nhãn. Với dữ liệu nhị phân, nhãn bình thường nên là `Normal`, `BENIGN`, `0` hoặc `Normal.`; các nhãn còn lại được coi là tấn công.

## Cài đặt trên Windows
Mở CMD/PowerShell trong thư mục dự án:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Sau đó mở địa chỉ local mà Streamlit hiển thị trong terminal (thường là http://localhost:8501).

## Cách dùng
1. Tải CSV từ sidebar.
2. Chọn cột nhãn.
3. Chọn tỷ lệ Test và số cây Random Forest.
4. Nhấn **Huấn luyện và đánh giá**.
5. Xem Accuracy, Precision, Recall, F1, confusion matrix và permutation importance.
6. Có thể tải một CSV mới có cùng các cột đặc trưng để dự đoán và tải kết quả về.

## Các điểm kỹ thuật
- Chia Train/Test trước khi fit imputer và One-Hot Encoder, dùng scikit-learn Pipeline để hạn chế data leakage.
- Xử lý giá trị thiếu: median cho cột số, most-frequent cho cột phân loại.
- One-Hot Encoding cho biến phân loại.
- `class_weight="balanced"` để điều chỉnh trọng số lớp trong Random Forest.
- Dùng Permutation Importance để có chỉ báo giải thích đơn giản; chưa tích hợp SHAP.
- Bản demo chuyển nhãn thành nhị phân: 0 = bình thường, 1 = tấn công.

## Giới hạn quan trọng
- Đây là bản demo học tập, chưa tích hợp Snort/Suricata, Kafka, SIEM/SOAR, PCAP trực tiếp hoặc luồng thời gian thực.
- Bản demo chưa có CNN+GRU; cần dữ liệu chuỗi được tạo đúng thứ tự thời gian để triển khai mô hình đó.
- Nếu bộ dữ liệu có nhiều nhãn nhưng không có nhãn bình thường nhận diện được, hãy chuẩn hóa nhãn trước.
- Các bộ dữ liệu benchmark khác nhau về cột và nhãn. Cần làm sạch và chuẩn hóa cột nhãn cho từng dataset.
- Chia ngẫu nhiên có thể tạo kết quả lạc quan nếu các flow liên quan hoặc trùng lặp xuất hiện ở cả Train và Test. Với nghiên cứu nghiêm túc, hãy cân nhắc chia theo thời gian/host/phiên và báo cáo rõ phương pháp.
- Không coi Accuracy cao là đủ; hãy xem Recall, Precision, F1 và confusion matrix.
- Không dùng để tự động chặn lưu lượng trong mạng sản xuất nếu chưa kiểm thử, rà soát bảo mật và có cơ chế phê duyệt.

## Cấu trúc
- `app.py`: ứng dụng dashboard Streamlit
- `requirements.txt`: thư viện cần cài
- `README.md`: hướng dẫn cài đặt và chạy
