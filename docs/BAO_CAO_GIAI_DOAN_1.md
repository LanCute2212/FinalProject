# Báo cáo giai đoạn 1: đánh giá mô hình cảm xúc

## Phạm vi đã làm

- Chỉ dùng các nhãn có nguồn gán thủ công và qua bước audit trước khi huấn luyện.
- So sánh baseline lớp đa số với Logistic Regression, Linear SVM, Naïve Bayes và TF-IDF word/character.
- Bổ sung nested cross-validation: inner 3-fold chọn `C` và `class_weight` theo macro-F1; outer 5-fold ước lượng quy trình chọn mô hình.
- Huấn luyện lại một ứng viên Logistic Regression bằng toàn bộ nhãn đã audit. Metadata lưu fingerprint của dữ liệu, tham số, protocol và trạng thái triển khai.
- Model Lab hiển thị phân bố lớp, chỉ số theo lớp, confusion matrix và cảnh báo khi không có test độc lập. Ứng viên hiện chưa bật để dự đoán trong website.

## Dữ liệu

Tập manual có 272 dòng; audit giữ lại 266 mẫu và đưa 3 mẫu vào diện quarantine, đồng thời nhận diện các dòng trùng nhất quán và xung đột với dữ liệu tham chiếu. Phân bố nhãn dùng để huấn luyện là 222 tích cực, 22 tiêu cực và 22 trung tính. Khoảng 85,7% mẫu thuộc về một sản phẩm, nên khả năng khái quát sang sản phẩm khác vẫn chưa được chứng minh.

File `manual_labels_test.csv` có nhãn cũ nhưng bị trùng/chéo với tập huấn luyện; audit xác định đây không phải test độc lập và không đưa các dòng này vào tập train.

## Kết quả hiện tại

Nested 5-fold outer / 3-fold inner trên 266 mẫu cho kết quả phát triển sau:

| Chỉ số | Trung bình ± độ lệch chuẩn |
| --- | ---: |
| Accuracy | 0,771 ± 0,041 |
| Macro Precision | 0,545 ± 0,114 |
| Macro Recall | 0,590 ± 0,123 |
| Macro F1 | 0,534 ± 0,074 |
| Recall tiêu cực | 0,590 ± 0,305 |
| Recall trung tính | 0,350 ± 0,308 |
| Recall tích cực | 0,829 ± 0,084 |

Điểm số này là ước lượng cross-validation, không phải kết quả trên test độc lập. Recall trung tính thấp và dao động lớn; do đó model được lưu như ứng viên nghiên cứu (`deployment_ready: false`) và không được dùng để tự động gán nhãn trong luồng website.

## Dữ liệu Tiki mới

Đã lấy chọn lọc hai file raw từ `origin/main`, không merge các thay đổi mã nguồn remote. Tập Tiki có 20.144 dòng, trong đó 9.537 dòng có bình luận không rỗng và 8.580 nội dung khác nhau. Sau khi gộp thêm 500 review mẫu và chạy lại pipeline hiện tại, dữ liệu processed có 20.644 dòng, 9.279 dòng đủ điều kiện phân tích văn bản, 10.645 dòng bị đánh dấu spam/rỗng và 727 dòng trùng. Số dòng không tự xác nhận tính chân thực của review; cần kiểm tra nguồn và lấy mẫu thủ công.

## Chạy lại

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sentiment.py
```

Kết quả được lưu tại `data/processed/evaluation/`; metadata và model ứng viên được tạo tại `models/` (thư mục này đang được `.gitignore` bỏ qua).

## Việc tiếp theo

1. Gán thêm nhãn cho nhận xét tiêu cực và trung tính, đồng thời lấy mẫu từ nhiều sản phẩm khác nhau; người gán không xem rating khi quyết định sentiment.
2. Tạo tập test manual độc lập theo sản phẩm, không trùng review hoặc nội dung với train.
3. Chạy lại nested evaluation, kiểm tra confusion matrix, recall từng lớp và các nhóm câu trái chiều/ít thông tin.
4. Chỉ bật dự đoán tự động trong dashboard sau khi test độc lập đạt tiêu chí đã thống nhất.
