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
│   └── utils/
│       └── logger.py            # Logger định dạng chuẩn
├── scripts/
│   ├── generate_sample_data.py  # Sinh dữ liệu mẫu giả lập chuẩn TMĐT (500 reviews, 5 SKU)
│   ├── run_crawl.py             # CLI cào dữ liệu từ Tiki / Shopee
│   └── run_preprocess.py        # CLI chạy tiền xử lý & in báo cáo thống kê định lượng
├── requirements.txt
└── README.md
```

---

## 2. Hướng dẫn sử dụng nhanh

Môi trường đã được cấu hình sẵn tại `.venv` (Python 3.11).

### 2.1. Sinh dữ liệu mẫu & chạy thử nghiệm ngay lập tức

```powershell
# Bước 1: Sinh 500 review mẫu thực tế (5 SKU của 1 shop công nghệ)
& ".\.venv\Scripts\python.exe" "scripts\generate_sample_data.py"

# Bước 2: Chạy pipeline tiền xử lý và in báo cáo định lượng
& ".\.venv\Scripts\python.exe" "scripts\run_preprocess.py"
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
*(File CSV chỉ cần có cột nội dung đánh giá: `comment_text` hoặc `review_text`)*

---

## 3. Các quy tắc nghiệp vụ (Business Rules) được đảm bảo trong Pipeline
* **BR10:** Kiểm soát điểm rating nằm nghiêm ngặt trong thang `[1..5]`.
* **BR11 & NFR19:** Bảo toàn 100% nội dung gốc trong cột `comment_text_raw`. Dữ liệu sạch lưu tại `comment_text_clean` và `comment_text_segmented`.
* **BR12:** Tự động tính mã băm MD5 để nhận diện các review trùng lặp nội dung (`is_duplicate = True`), không tính nhiều lần vào thống kê.
* **BR09 & NFR16:** Tự động lọc bỏ các review săn xu vô nghĩa (`COIN_FARMING_SPAM`), các câu gợi ý tự động của sàn (`DEFAULT_PLATFORM_TEMPLATE`), và các chuỗi ký tự lặp rác (`REPETITIVE_CHAR_NOISE`).
