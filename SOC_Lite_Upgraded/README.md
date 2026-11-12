# SOC Lite ML-IDS — bản nâng cấp

Giữ lại giao diện 3 tab của dự án, bổ sung **2 tính năng chính**:

1. Nạp dữ liệu **NSL-KDD thật** (`KDDTrain+.txt`, `KDDTest+.txt`), 41 đặc trưng, chuyển nhãn tấn công sang `Normal/DoS/Probe/R2L/U2R`, giữ nguyên tập kiểm thử chuẩn. Nạp dữ liệu **CICIDS2017 thật** từ một hoặc nhiều CSV `MachineLearningCSV`. Giữ nhãn gốc của CICIDS2017, không ép sang nhãn của NSL-KDD.
2. Tiền xử lý đúng quy trình **fit trên Train**, sau đó chỉ **transform trên Test**: thay NaN và inf, dùng trung vị/mốt học từ Train, One-Hot Encoding với giá trị mới được bỏ qua an toàn, MinMaxScaler hoặc StandardScaler, chọn lọc Top-K đặc trưng bằng ANOVA F (học trên Train).

**Không kèm dữ liệu benchmark** (người dùng cần tải các file thực), và ứng dụng **không bắt gói tin Wi-Fi trực tiếp**.

## Cài đặt / chạy bằng CMD (Windows)

1. Giải nén cả thư mục `SOC_Lite_Upgraded`.
2. Mở CMD trong thư mục chứa `app.py` (hoặc `cd /d DUONG_DAN_DEN_THU_MUC`).
3. Chạy:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Mở `http://localhost:8501`.

## Dữ liệu lấy từ đâu?

- **NSL-KDD**: file gốc `KDDTrain+.txt` và `KDDTest+.txt` (43 trường: 41 đặc trưng, nhãn, difficulty). Trang giới thiệu của UNB: https://www.unb.ca/cic/datasets/nsl.html . **UNB hiện ghi rõ không còn cung cấp bản tải trực tiếp**, nên bạn cần lấy bản phân phối/đối chiếu đáng tin cậy và ghi nguồn rõ trong báo cáo.
- **CICIDS2017**: bản gốc `MachineLearningCSV.zip` chứa các file `.csv` có cột `Label` và `Flow Duration`. Trang chính thức: https://www.unb.ca/cic/datasets/ids-2017.html

### Cách nạp trong ứng dụng

**NSL-KDD**: Sidebar > `NSL-KDD thật` > tải `KDDTrain+.txt` và `KDDTest+.txt` > `Nạp NSL-KDD thật`.

**CICIDS2017**: Sidebar > `CICIDS2017 thật` > chọn 1 hoặc nhiều file `.csv` đã giải nén > `Nạp CICIDS2017 thật`. Bộ này có nhiều file dung lượng lớn; nên chọn nhiều ngày có nhãn `BENIGN` và tấn công. Mỗi file được lấy mẫu ngẫu nhiên tối đa N dòng (mặc định 20.000) để chạy được trên máy phổ thông. **Kết quả trên tập lấy mẫu không phải kết quả của toàn bộ CICIDS2017**.

Nếu file vượt giới hạn tải lên, đặt file tại các thư mục bên dưới hoặc chọn đường dẫn riêng trên sidebar:

```text
SOC_Lite_Upgraded/
  data/
    nsl_kdd/KDDTrain+.txt
    nsl_kdd/KDDTest+.txt
    cicids2017/*.csv
```

Cách `Đọc file từ thư mục máy` cho phép đọc file lớn mà không cần qua trình duyệt. Nên chạy CMD tại chính thư mục dự án để các đường dẫn tương đối hoạt động đúng.

## Luồng xử lý

`File thực → kiểm tra định dạng & nhãn → chia Train/Test (riêng NSL giữ split gốc) → fit imputer/OneHot/scaler/Top-K trên Train → transform Test → huấn luyện 3 mô hình → đánh giá → dự đoán mẫu Test → JSON cảnh báo`.

- One-Hot `handle_unknown="ignore"`: các giá trị mới của một cột phân loại không khiến ứng dụng bị lỗi.
- Chọn Top-K dựa trên ANOVA F với nhãn Train; có thể tắt để giữ nguyên mọi đặc trưng đã mã hóa.
- Với CICIDS2017, các cột nhận dạng như IP, Flow ID, Timestamp được loại bỏ để giảm rủi ro học thuộc dữ liệu nhận dạng.
- Với CICIDS2017, một số nhãn tấn công khác `Normal/DoS/Probe/R2L/U2R`. Mô-đun cảnh báo đang trả mặc định `MEDIUM/LOG_ONLY` cho nhãn chưa ánh xạ; chưa có phân loại mức độ nguy hiểm riêng theo CICIDS2017.
- Nếu bạn chỉ tải một ngày dữ liệu CICIDS2017 và ngày đó có một lớp nhãn, ứng dụng sẽ yêu cầu nạp thêm file để có ít nhất 2 nhãn.
- **Không so sánh trực tiếp điểm mô hình được huấn luyện trên hai bộ dữ liệu có đặc trưng/nhãn khác nhau** nếu chưa thiết kế một quy trình đánh giá hợp lệ.

## Chạy kiểm thử logic

```cmd
python -m unittest discover -s tests -v
```

Các test tự tạo một vài dòng **giả lập cấu trúc** (không phải dữ liệu benchmark thật) để xác minh đường đọc, chia nhãn và xử lý Train/Test. Muốn chứng minh tiêu chí niên luận, cần nạp đúng tập dữ liệu thật và lưu kết quả thực nghiệm.
