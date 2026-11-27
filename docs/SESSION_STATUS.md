# Trạng thái phiên làm việc

> Cập nhật lần cuối: 2026-11-27. File này dùng để bàn giao trạng thái dự án khi mở chat Codex mới; hãy kiểm tra working tree trước khi chỉnh sửa.

## Mục tiêu dự án

Xây dựng website Streamlit phân tích đánh giá sản phẩm thương mại điện tử, ưu tiên theo thứ tự:

1. Phân loại sentiment bình luận thành tích cực, tiêu cực, trung tính với độ tin cậy được đánh giá đúng.
2. Nhận diện aspect trong bình luận và sentiment riêng của từng aspect.
3. Hoàn thiện dashboard và biểu đồ sau khi các luồng phân tích cốt lõi, dữ liệu và bug đã ổn định.

## Trạng thái repo hiện tại

- Thư mục làm việc: `E:\Đồ án`.
- Nhánh: `main`, đang chậm hơn `origin/main` 8 commit tại lần kiểm tra gần nhất.
- Working tree có nhiều sửa đổi chưa commit ở `app.py`, dữ liệu, pipeline tiền xử lý, scripts và tests. Có các file mới chưa theo dõi trong `src/modeling/`, `src/data_quality.py`, `scripts/`, `tests/` và `docs/BAO_CAO_GIAI_DOAN_1.md`.
- `docs/TAI_LIEU_ON_TAP_DO_AN.txt` hiện là file bị xóa trong working tree. Kiểm tra trước khi khôi phục hoặc commit.
- Dữ liệu Tiki mới được lấy chọn lọc từ remote; chưa merge 8 commit code còn thiếu. Không reset, checkout hàng loạt, rebase hay merge trước khi xem `git status` và xác nhận các thay đổi local được bảo toàn.

## Đã thực hiện

### Dữ liệu và tiền xử lý

- Pipeline hiện có các bước làm sạch, chuẩn hóa tiếng Việt/teencode, phát hiện spam và trùng lặp, và tạo dữ liệu processed.
- Theo báo cáo giai đoạn 1, dữ liệu Tiki raw có 20.144 dòng, 9.537 bình luận không rỗng, 8.580 nội dung bình luận khác nhau.
- Lần tiền xử lý được ghi nhận tạo `data/processed/reviews_processed.csv` gồm 20.644 dòng: 9.279 dòng đủ điều kiện phân tích văn bản, 10.645 dòng bị đánh dấu spam/rỗng và 727 dòng trùng. Các con số này chưa xác nhận bình luận là dữ liệu thật; cần kiểm tra nguồn và lấy mẫu thủ công.

### Giai đoạn 1: đánh giá mô hình sentiment

- Báo cáo chi tiết: [`BAO_CAO_GIAI_DOAN_1.md`](BAO_CAO_GIAI_DOAN_1.md).
- Đã audit nhãn thủ công và so sánh baseline lớp đa số với Logistic Regression, Linear SVM, Naïve Bayes trên TF-IDF word/character.
- Nested cross-validation: 5 outer folds, 3 inner folds; chọn tham số theo macro-F1.
- Bộ train sau audit có 266 mẫu: 222 tích cực, 22 tiêu cực, 22 trung tính. Khoảng 85,7% mẫu tập trung ở một sản phẩm, nên khả năng khái quát sang sản phẩm khác chưa được chứng minh.
- `manual_labels_test.csv` không phải test độc lập do trùng/chéo dữ liệu với train; không dùng file này làm test độc lập hoặc bổ sung nhãn train.
- Kết quả nested CV: accuracy `0.771 ± 0.041`, macro precision `0.545 ± 0.114`, macro recall `0.590 ± 0.123`, macro-F1 `0.534 ± 0.074`; recall tiêu cực `0.590 ± 0.305`, trung tính `0.350 ± 0.308`, tích cực `0.829 ± 0.084`.
- Ứng viên Logistic Regression được lưu với `deployment_ready: false`. Website chưa được bật để tự động gán sentiment bằng ứng viên này; accuracy tổng thể chưa đủ để kết luận mô hình tốt khi lớp bị lệch và recall lớp trung tính thấp.
- Model Lab trong `app.py` hiển thị phân bố lớp, chỉ số theo lớp, confusion matrix và cảnh báo thiếu test độc lập.

### Aspect-based sentiment (ABSA)

- Hiện nhận diện aspect chủ yếu bằng từ điển offline `data/dictionaries/aspect_keywords.json`.
- Chưa có nhãn gold đủ dùng và mô hình đã đánh giá để gắn sentiment riêng cho từng aspect trong câu nhiều khía cạnh.
- Mục tiêu phân tích 7 aspect: chất lượng, giá cả, thiết kế, độ bền, đóng gói, giao hàng và chăm sóc khách hàng.

## Kiểm tra gần nhất

- Ngày 2026-11-27: `.\.venv\Scripts\python.exe -m unittest discover -s tests -q` — 36 tests passed.
- Cùng ngày, health endpoint của Streamlit tại `http://localhost:8501/_stcore/health` trả `ok`. Server có thể đã dừng khi bắt đầu phiên mới.
- Chạy ứng dụng: `.\.venv\Scripts\python.exe -m streamlit run app.py`.
- Chạy đánh giá sentiment: `.\.venv\Scripts\python.exe scripts/evaluate_sentiment.py`.
- Chạy lại preprocessing: `.\.venv\Scripts\python.exe scripts/run_preprocess.py --raw_dir data/raw --output data/processed/reviews_processed.csv`.

## Việc tiếp theo — ưu tiên

1. **Tiếp tục giai đoạn 1:** bổ sung nhãn thủ công cho sentiment tiêu cực và trung tính, lấy mẫu từ nhiều sản phẩm; người gán nhãn không xem rating khi quyết định sentiment.
2. Tạo tập test thủ công, độc lập theo sản phẩm; loại bỏ trùng lặp văn bản và review giữa train/validation/test.
3. Đánh giá lại Logistic Regression, SVM và Naïve Bayes bằng macro-F1, macro-recall, recall từng lớp, confusion matrix và sai số theo sản phẩm. Đặt tiêu chí nghiệm thu trước khi thử deploy.
4. Sau khi có dữ liệu sentiment đáng tin cậy, xây dựng nhãn aspect và sentiment theo aspect; đánh giá nhận diện aspect và phân cực riêng từng aspect.
5. Sửa các bug còn lại rồi mới mở rộng dashboard/biểu đồ theo yêu cầu đề tài.
6. Đối chiếu 8 commit mới trên `origin/main` theo từng file/diff; chỉ tích hợp phần cần thiết, không ghi đè các thay đổi local.

## Lưu ý khi tiếp tục ở chat mới

- Bắt đầu bằng `git status --short --branch` và đọc file này cùng báo cáo giai đoạn 1.
- Giữ nguyên dữ liệu thủ công và thay đổi local hiện có; không dùng rating làm nhãn gold sentiment.
- Không dùng `manual_labels_test.csv` làm test độc lập.
- Không bật ứng viên mô hình cho suy luận production cho tới khi có test độc lập và đạt tiêu chí đánh giá đã thống nhất.
