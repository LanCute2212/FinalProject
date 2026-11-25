# Customer Voice Analytics — Data Engineering & Preprocessing Pipeline

Hệ thống phân tích và trực quan hóa cảm xúc khách hàng từ đánh giá sản phẩm trên sàn thương mại điện tử (Dành cho 1 Seller - Đa sản phẩm).

Đồ án tốt nghiệp Học viện Công nghệ Bưu chính Viễn thông (PTIT) - Nhóm 03.

---

## 1. Cấu trúc thư mục

```
e:\Đồ án\
├── data/
│   ├── dictionaries/            # Từ điển hỗ trợ NLP tiếng Việt
│   │   ├── teencode_dict.json   # Ánh xạ từ viết tắt, teencode TMĐT
│   │   ├── emoji_dict.json      # Phân loại và token hóa Emoji
│   │   └── spam_patterns.json   # Mẫu nhận diện câu mặc định sàn & spam săn xu
│   ├── raw/                     # Dữ liệu gốc cào về (CSV, JSON) - Bảo toàn 100% (BR11)
│   └── processed/               # Dữ liệu sạch sau lọc spam, trùng lặp & tách từ
├── src/
│   ├── crawler/                 # Module cào dữ liệu TMĐT
│   │   ├── base_crawler.py      # Base class tuân thủ BR04, BR05, BR06, BR10, BR11
│   │   ├── tiki_crawler.py      # Cào review sản phẩm qua API Tiki
│   │   └── shopee_crawler.py    # Cào review sản phẩm qua API Shopee
│   ├── preprocessing/           # Module tiền xử lý dữ liệu tiếng Việt
│   │   ├── cleaner.py           # Làm sạch HTML, URL, kéo dài ký tự, chuẩn hóa NFC
│   │   ├── normalizer.py        # Chuẩn hóa teencode, ánh xạ emoji, dấu câu
│   │   ├── spam_detector.py     # Lọc review săn xu, câu mặc định & review trùng lặp (BR12)
│   │   ├── tokenizer.py         # Phân đoạn từ tiếng Việt (underthesea / pyvi)
│   │   └── pipeline.py          # Pipeline tổng hợp xuất DataFrame
│   ├── storage/
│   │   └── sql_server_store.py  # Lưu products, reviews và crawl_runs
│   └── utils/
│       └── logger.py            # Logger định dạng chuẩn
├── scripts/
│   ├── generate_sample_data.py  # Sinh dữ liệu mẫu giả lập chuẩn TMĐT (500 reviews, 5 SKU)
│   ├── run_crawl.py             # CLI cào dữ liệu từ Tiki / Shopee
│   ├── import_csv_to_sql_server.py # Import review từ CSV vào SQL Server
│   ├── export_sql_server_to_csv.py # Export review SQL Server sang CSV
│   ├── run_eda.py               # EDA rating, độ dài, nghịch lý sao-text, aspect coverage
│   ├── run_features.py          # Tạo nhãn sentiment, aspect labels và TF-IDF baseline
│   └── run_preprocess.py        # CLI chạy tiền xử lý & in báo cáo thống kê định lượng
├── requirements.txt
└── README.md
```

---

## 2. Hướng dẫn sử dụng nhanh

Môi trường đã được cấu hình sẵn tại `.venv` (Python 3.11).
Các danh sách chọn trong dashboard có ô tìm kiếm ngay bên dưới; ô tìm kiếm hỗ trợ từ khóa tiếng Việt không dấu.

### 2.1. Sinh dữ liệu mẫu & chạy thử nghiệm ngay lập tức

```powershell
# Bước 1: Sinh 500 review mẫu thực tế (5 SKU của 1 shop công nghệ)
& ".\.venv\Scripts\python.exe" "scripts\generate_sample_data.py"

# Bước 2: Chạy pipeline tiền xử lý và in báo cáo định lượng
& ".\.venv\Scripts\python.exe" "scripts\run_preprocess.py"

# Bước 3: Phân tích khám phá dữ liệu (EDA)
& ".\.venv\Scripts\python.exe" "scripts\run_eda.py"

# Bước 4: Tạo feature cho sentiment/aspect baseline
& ".\.venv\Scripts\python.exe" "scripts\run_features.py"
```

### 2.2. Cào dữ liệu thực tế từ sàn TMĐT

#### Cào từ Tiki:
```powershell
# Cào 100 review cho 1 sản phẩm Tiki cụ thể (truyền product_id)
& ".\.venv\Scripts\python.exe" "scripts\run_crawl.py" --platform tiki --product_id 197455850 --max_reviews 100
```

#### Cào từ Shopee:
```powershell
# Cào review từ Shopee (cần shop_id và item_id)
& ".\.venv\Scripts\python.exe" "scripts\run_crawl.py" --platform shopee --shop_id 88204689 --product_id 192837465 --product_name "Tên sản phẩm" --max_reviews 100
```

### 2.3. Tiền xử lý file dữ liệu nhóm tự chuẩn bị (Custom CSV)
Nếu nhóm đã có file CSV tự cào trước đó, chỉ cần chạy:
```powershell
& ".\.venv\Scripts\python.exe" "scripts\run_preprocess.py" --input "đường_dẫn_file_của_nhóm.csv" --output "data\processed\reviews_processed.csv"
```
*(File CSV đầu vào cần có cột nội dung đánh giá `comment_text`.)*

### 2.4. SQL Server (CSV vẫn được giữ)

SQL Server lưu riêng thông tin sản phẩm (`products`), từng review (`reviews`) và mỗi lần chạy crawler (`crawl_runs`). Review được định danh theo bộ `platform + product_id + review_id`; crawl và tiền xử lý vẫn giữ các file CSV/JSON hiện tại. Máy cần có SQL Server đang chạy và ODBC Driver 17 for SQL Server. Trước khi chạy, dùng SSMS bằng tài khoản quản trị SQL Server để:

1. Kết nối SSMS tới đúng SQL Server instance đã cấu hình ở `SQLSERVER_SERVER` (mặc định `localhost`) và tạo database `customer_voice_analytics`.
2. Trong PowerShell chạy ứng dụng, dùng `whoami` để xác định Windows principal; thêm đúng principal đó thành server login trong **Security → Logins** (không chỉ tạo database user).
3. Trong **User Mapping**, ánh xạ login vào `customer_voice_analytics` với role `db_owner`.

Ứng dụng dùng Windows Authentication; role `db_owner` chỉ cần trên database này để khởi tạo bảng/index và đọc/ghi dữ liệu. Không cần gửi mật khẩu qua ứng dụng.

```powershell
$env:SQLSERVER_SERVER = "localhost"
$env:SQLSERVER_DATABASE = "customer_voice_analytics"
$env:SQLSERVER_DRIVER = "ODBC Driver 17 for SQL Server"
```

Sau khi cài dependencies từ `requirements.txt`, có thể import bộ processed CSV sẵn có hoặc export reviews về CSV:

```powershell
& ".\.venv\Scripts\python.exe" "scripts\import_csv_to_sql_server.py"
& ".\.venv\Scripts\python.exe" "scripts\export_sql_server_to_csv.py" --output "data\exports\reviews.csv"
```

Khi đã cấu hình `SQLSERVER_SERVER`, CLI crawler và preprocessing sẽ đồng bộ review sang SQL Server đồng thời vẫn ghi CSV/JSON. Dùng `--csv_only` để chạy riêng bằng CSV. Với crawler trên Dashboard, chọn **SQL Server** trong **Cài đặt nâng cao** trước khi crawl để lưu lịch sử vào `crawl_runs` và đồng bộ review; nếu chọn CSV thì lần crawl chỉ lưu file. Dashboard cũng dùng lựa chọn này để đọc dữ liệu. Lỗi kết nối SQL Server sẽ được báo rõ, không tự chuyển nguồn dữ liệu.

### 2.5. Gán nhãn thủ công để chuẩn bị tập đánh giá

Khởi động Dashboard:
```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Trong **Phòng Nghiên cứu & Mô hình AI** → **Công cụ Gán nhãn Dữ liệu Thủ công**, chọn review và gán sentiment/aspect theo nội dung. Nhãn được lưu tại `data/processed/manual_labels.csv`, phân biệt review bằng nền tảng và mã review; có thể tải CSV từ giao diện. Nên gán nhãn cân bằng giữa ba lớp cảm xúc và không dùng nhãn suy ra từ rating làm nhãn đánh giá độc lập.

### 2.6. Khám phá và duyệt từ khóa Aspect

Trong **Phân tích Khía cạnh Sản phẩm (ABSA)** → **Khám phá từ khóa Aspect**, hệ thống gom các unigram/bigram/trigram chưa có trong `aspect_keywords.json` theo số review hợp lệ có chứa cụm từ. Xem review mẫu, chọn aspect hiện có hoặc tạo mã aspect mới; keyword chỉ được thêm sau khi xác nhận. Candidate bị bỏ qua được lưu tại `data/processed/aspect_candidates_ignored.json`. Đây là gợi ý dựa trên tần suất, không phải kết quả phân loại ngữ nghĩa tự động.

---

## 3. Các quy tắc nghiệp vụ (Business Rules) được đảm bảo trong Pipeline
* **BR10:** Kiểm soát điểm rating nằm nghiêm ngặt trong thang `[1..5]`.
* **BR11 & NFR19:** Bảo toàn 100% nội dung gốc trong cột `comment_text_raw`. Dữ liệu sạch lưu tại `comment_text_clean` và `comment_text_segmented`.
* **BR12:** Tự động tính mã băm MD5 để nhận diện các review trùng lặp nội dung (`is_duplicate = True`), không tính nhiều lần vào thống kê.
* **BR09 & NFR16:** Tự động lọc bỏ các review săn xu vô nghĩa (`COIN_FARMING_SPAM`), các câu gợi ý tự động của sàn (`DEFAULT_PLATFORM_TEMPLATE`), và các chuỗi ký tự lặp rác (`REPETITIVE_CHAR_NOISE`).
