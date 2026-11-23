# TÀI LIỆU ÔN TẬP VÀ ĐỊNH HƯỚNG PHÁT TRIỂN ĐỒ ÁN TỐT NGHIỆP

**ĐỀ TÀI:** XÂY DỰNG HỆ THỐNG PHÂN TÍCH VÀ TRỰC QUAN HÓA CẢM XÚC KHÁCH HÀNG TỪ ĐÁNH GIÁ SẢN PHẨM TRÊN SÀN THƯƠNG MẠI ĐIỆN TỬ  
**ĐƠN VỊ:** HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT)  
**NHÓM THỰC HIỆN:** NHÓM 03 (LONG & MẠNH)  
**NGÀY CẬP NHẬT:** 12/11/2026  

---

## MỤC LỤC TỔNG QUAN
1. [TỔNG QUAN ĐỀ TÀI & PHÂN CÔNG TRÁCH NHIỆM](#1-tổng-quan-đề-tài--phân-công-trách-nhiệm)
2. [ĐỐI CHIẾU HIỆN TRẠNG VỚI YÊU CẦU CỦA GIẢNG VIÊN HƯỚNG DẪN](#2-đối-chiếu-hiện-trạng-với-yêu-cầu-của-giảng-viên-hướng-dẫn)
3. [DANH SÁCH HẠNG MỤC CẦN NÂNG CẤP CHUYÊN SÂU (ADVANCED)](#3-danh-sách-hạng-mục-cần-nâng-cấp-chuyên-sâu-advanced)
4. [QUY TRÌNH THỰC HIỆN (WORKFLOW END-TO-END)](#4-quy-trình-thực-hiện-workflow-end-to-end)
5. [CẨM NANG TOÀN TẬP LÝ THUYẾT BẮT BUỘC PHẢI NẮM VỮNG](#5-cẩm-nang-toàn-tập-lý-thuyết-bắt-buộc-phải-nắm-vững)
   - [5.1. Lý thuyết Xử lý Ngôn ngữ Tự nhiên tiếng Việt (Vietnamese NLP)](#51-lý-thuyết-xử-lý-ngôn-ngữ-tự-nhiên-tiếng-việt-vietnamese-nlp)
   - [5.2. Lý thuyết Biểu diễn Văn bản (Text Representation & Feature Engineering)](#52-lý-thuyết-biểu-diễn-văn-bản-text-representation--feature-engineering)
   - [5.3. Lý thuyết Các Thuật toán Học máy & Học sâu (Machine Learning & Deep Learning)](#53-lý-thuyết-các-thuật-toán-học-máy--học-sâu-machine-learning--deep-learning)
   - [5.4. Lý thuyết Phân tích Cảm xúc theo Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)](#54-lý-thuyết-phân-tích-cảm-xúc-theo-khía-cạnh-aspect-based-sentiment-analysis---absa)
   - [5.5. Lý thuyết Đánh giá & Kiểm thử Mô hình (Evaluation Metrics)](#55-lý-thuyết-đánh-giá--kiểm-thử-mô-hình-evaluation-metrics)
   - [5.6. Lý thuyết Thiết kế Cơ sở Dữ liệu Quan hệ (RDBMS)](#56-lý-thuyết-thiết-kế-cơ-sở-dữ-liệu-quan-hệ-rdbms)
6. [BỘ CÂU HỎI VẤN ĐÁP THƯỜNG GẶP KHI BẢO VỆ ĐỒ ÁN (VÀ GỢI Ý TRẢ LỜI)](#6-bộ-câu-hỏi-vấn-đáp-thường-gặp-khi-bảo-vệ-đồ-án-và-gợi-ý-trả-lời)

---

## 1. TỔNG QUAN ĐỀ TÀI & PHÂN CÔNG TRÁCH NHIỆM

### 1.1. Mục tiêu đề tài
Xây dựng một hệ thống hoàn chỉnh có khả năng:
1. Thu thập dữ liệu đánh giá, bình luận của khách hàng từ các sàn thương mại điện tử (Shopee, Tiki,...).
2. Tiền xử lý, chuẩn hóa dữ liệu tiếng Việt đặc thù của thương mại điện tử (teencode, emoji, lỗi chính tả, spam).
3. Tự động phân loại cảm xúc tổng quan của bình luận (Tích cực / Tiêu cực / Trung tính).
4. Phân tích cảm xúc đa khía cạnh (Aspect-based Sentiment Analysis): nhận diện khách hàng đang đánh giá về khía cạnh nào (Chất lượng, Giá cả, Thiết kế, Độ bền, Đóng gói, Giao hàng, CSKH) và sắc thái cảm xúc tương ứng trên từng khía cạnh.
5. Trực quan hóa dữ liệu trên Dashboard quản trị giúp nhà bán hàng (Seller) nắm bắt phản hồi, phát hiện lỗi sản phẩm/vận hành và ra quyết định kinh doanh.

### 1.2. Phân công vai trò trong nhóm
- **Thành viên 1 (Long):** Phụ trách Data Engineering, Pipeline Tiền xử lý, Thiết kế CSDL và Giao diện Dashboard Trực quan hóa (Phase 1, 4, 5, 8).
- **Thành viên 2 (Mạnh):** Phụ trách Huấn luyện Mô hình AI (Machine Learning & Deep Learning), Tinh chỉnh siêu tham số và Đánh giá, So sánh kết quả thực nghiệm (Phase 6, 7).

---

## 2. ĐỐI CHIẾU HIỆN TRẠNG VỚI YÊU CẦU CỦA GIẢNG VIÊN HƯỚNG DẪN

Dựa trên tài liệu hướng dẫn của cô giáo (`HD De tai nhom Long Mạnh.pdf`), bảng dưới đây đối chiếu chi tiết giữa yêu cầu và những gì nhóm đã làm:

| STT | Chức năng theo yêu cầu của cô | Hiện trạng triển khai trong source code | Mức độ hoàn thành |
| :---: | :--- | :--- | :---: |
| **1** | **Thu thập dữ liệu đánh giá:**<br>- Tên sản phẩm, Danh mục sản phẩm<br>- Nội dung bình luận<br>- Số sao đánh giá (1-5★)<br>- Thời gian đánh giá<br>- Tên/mã người đánh giá<br>- Số lượt thích bình luận | - Đã có Crawler Tiki (`src/crawler/tiki_crawler.py`), Shopee (`shopee_crawler.py`) và bộ dữ liệu mẫu (`generate_sample_data.py`).<br>- Thu thập được: `review_id`, `product_id`, `product_name`, `rating`, `comment_text`, `created_at`, `raw_payload`.<br>- Đã cào thực tế hơn **1.200 review từ Tiki**.<br>*(Còn thiếu: lưu riêng cột `category` và `like_count`)* | **80%** |
| **2** | **Tiền xử lý dữ liệu:**<br>- Loại bỏ bình luận trống<br>- Loại bỏ dữ liệu trùng lặp<br>- Chuẩn hóa chữ hoa/thường<br>- Xử lý ký tự đặc biệt, emoji<br>- Xử lý từ viết tắt, lỗi chính tả<br>- Chuẩn hóa tiếng Việt (dấu, từ đơn/ghép) | - Hoàn thành xuất sắc tại `src/preprocessing/`:<br>+ `cleaner.py`: Xóa HTML, URL, kéo dài ký tự, chuẩn hóa NFC.<br>+ `normalizer.py`: Ánh xạ teencode (`teencode_dict.json`), emoji (`emoji_dict.json`).<br>+ `spam_detector.py`: Lọc spam mẫu mặc định, review săn xu, khử trùng lặp MD5 (BR12).<br>+ `tokenizer.py`: Phân đoạn từ tiếng Việt chuẩn bằng thư viện chuyên dụng `underthesea`. | **95%** |
| **3** | **Phân tích cảm xúc (Sentiment Analysis):**<br>- 3 nhóm: Tích cực / Tiêu cực / Trung tính<br>- Phương pháp: Naïve Bayes, SVM, LSTM, PhoBERT | - Hiện tại mới xây dựng **Baseline Heuristic**: suy luận tạm từ số sao rating (1-2★: Tiêu cực, 3★: Trung tính, 4-5★: Tích cực) và bộ quy tắc đối lập text-rating.<br>- Đã trích xuất đặc trưng TF-IDF sẵn sàng cho huấn luyện (`src/features/feature_engineering.py`).<br>*(Chưa train mô hình ML/DL thực thụ)* | **40%** *(Cần làm tiếp)* |
| **4** | **Phân tích theo khía cạnh (ABSA):**<br>- 7 khía cạnh: Chất lượng, Giá cả, Thiết kế, Độ bền, Đóng gói, Giao hàng, CSKH<br>- Bóc tách cặp: (Khía cạnh, Cảm xúc) | - Hiện tại mới dừng lại ở bước nhận diện khía cạnh bằng **từ điển từ khóa offline** (`aspect_keywords.json`).<br>- Chưa gắn được sắc thái cảm xúc cho từng khía cạnh riêng biệt khi một câu có nhiều khía cạnh trái chiều. | **35%** *(Cần làm tiếp)* |
| **5** | **Trực quan hóa dữ liệu (Dashboard):**<br>- Tổng lượt đánh giá, Tỷ lệ cảm xúc<br>- Biểu đồ theo sản phẩm, theo thời gian<br>- Top sản phẩm tích cực / tiêu cực nhất<br>- Từ khóa nhắc đến nhiều nhất<br>- Thống kê cảm xúc theo khía cạnh | - Đã dựng Dashboard nền tảng bằng **Streamlit** (`app.py`):<br>+ Bộ lọc đa sàn (Tiki, Shopee/Sample), bộ lọc sản phẩm, slider rating, checkbox review hợp lệ.<br>+ Thống kê số lượng, rating trung bình, tỷ lệ hợp lệ, spam.<br>+ Biểu đồ phân bố sao, chất lượng dữ liệu.<br>+ Bảng Review Explorer tìm kiếm nội dung.<br>+ Tab EDA kiểm tra nghịch lý sao-chữ, độ bao phủ khía cạnh.<br>*(Còn thiếu: Biểu đồ chuỗi thời gian, Leaderboard Top sản phẩm khen/chê, WordCloud)* | **65%** |
| **6** | **Thiết kế CSDL & Đánh giá mô hình** | - Dữ liệu đang lưu trữ dạng file `.csv` và `.json`.<br>- Chưa có Database quan hệ (SQLite/Postgres).<br>- Chưa có tập Test đánh giá các chỉ số Accuracy, Precision, Recall, F1-score. | **20%** *(Cần làm tiếp)* |

---

## 3. DANH SÁCH HẠNG MỤC CẦN NÂNG CẤP CHUYÊN SÂU (ADVANCED)

Để đồ án đạt điểm Giỏi/Xuất sắc và thuyết phục tuyệt đối hội đồng chấm, nhóm cần nâng cấp 4 hạng mục sau:

### 3.1. Nâng cấp Mô hình AI (Trọng tâm của bạn Mạnh)
1. **Chuyển dịch từ Heuristic sang Machine Learning thực thụ:**
   - Xây dựng 2 nhóm mô hình:
     - **Nhóm 1 (Baseline Models):** TF-IDF + Multinomial Naïve Bayes, TF-IDF + Linear Support Vector Machine (LinearSVC/SVM).
     - **Nhóm 2 (Advanced / Deep Learning Models):** Fine-tuning PhoBERT (phobert-base) hoặc xây dựng mạng Bi-LSTM kết hợp Word2Vec/FastText.
2. **Giải quyết bài toán Aspect-Based Sentiment Analysis (ABSA) chuẩn khoa học:**
   - Thay vì chỉ đếm từ khóa, xây dựng mô hình phân loại đa nhãn (Multi-label Classification) cho 7 khía cạnh:
     $$\text{Review} \xrightarrow{\text{Mô hình}} \{(A_1, S_1), (A_2, S_2), \dots\}$$
     Ví dụ: *"Máy xài mượt, pin trâu nhưng giao hàng quá chậm"* ➔ `(Chất lượng: Tích cực)`, `(Độ bền/Pin: Tích cực)`, `(Giao hàng: Tiêu cực)`.

### 3.2. Nâng cấp Hệ thống Dữ liệu & CSDL (Trọng tâm của bạn Long)
1. **Chuyển đổi từ File sang Cơ sở dữ liệu quan hệ (RDBMS):**
   - Sử dụng **SQLite** (nhẹ, không cần cài server phức tạp, dễ nộp kèm đồ án) hoặc **PostgreSQL**.
   - Thiết kế các bảng chuẩn hóa theo quan hệ 1-N: `sellers`, `products`, `reviews`, `aspects`, `review_aspect_predictions`.
2. **Bổ sung các trường thông tin theo đúng yêu cầu:**
   - Cập nhật crawler và database để thu thập thêm: `category` (Danh mục sản phẩm) và `thank_count` (Số lượt like bình luận).

### 3.3. Nâng cấp Giao diện Dashboard Quản trị (Trọng tâm của bạn Long)
Bổ sung các biểu đồ còn thiếu theo đúng trang 2 của tài liệu hướng dẫn:
1. **Biểu đồ cảm xúc theo chuỗi thời gian (Sentiment Trend over Time):** Biểu đồ đường (Line chart) thể hiện số lượng review tích cực/tiêu cực biến động theo tuần/tháng.
2. **Bảng xếp hạng (Leaderboard):**
   - Top 5 sản phẩm được đánh giá hài lòng nhất (Tỷ lệ tích cực cao nhất).
   - Top 5 sản phẩm cần cảnh báo (Nhiều phản ánh tiêu cực nhất).
3. **Đám mây từ khóa (WordCloud) & Top n-grams:** Hiển thị trực quan những từ ngữ/cụm từ xuất hiện nhiều nhất trong tập đánh giá tiêu cực (để người bán biết lỗi ở đâu).
4. **Trang Dự đoán Review Mới (Live Inference):** Cho phép người dùng gõ hoặc dán một câu bình luận bất kỳ vào ô nhập liệu, bấm nút "Dự đoán" ➔ Hệ thống gọi mô hình AI để trả về ngay lập tức nhãn cảm xúc và từng khía cạnh liên quan.

### 3.4. Kiểm thử & Đánh giá Thực nghiệm (Cả 2 bạn phối hợp)
1. Chia tập dữ liệu chuẩn thành 3 phần độc lập: **Train (70%) - Validation (15%) - Test (15%)**.
2. Lập bảng so sánh hiệu năng chi tiết giữa các mô hình (Naïve Bayes vs SVM vs Bi-LSTM vs PhoBERT) trên các độ đo: **Accuracy, Macro-Precision, Macro-Recall, Macro-F1**.
3. Vẽ **Ma trận nhầm lẫn (Confusion Matrix)** để phân tích các trường hợp mô hình đoán sai.

---

## 4. QUY TRÌNH THỰC HIỆN (WORKFLOW END-TO-END)

Toàn bộ hệ thống vận hành theo quy trình khép kín 6 giai đoạn:

```
[Sàn TMĐT: Tiki / Shopee]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: THU THẬP & LƯU TRỮ RAW DATA              │
│ - Tiki API / Shopee Crawler                            │
│ - Lưu trữ thô bảo toàn nguyên vẹn (data/raw/)          │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: PIPELINE TIỀN XỬ LÝ DỮ LIỆU TIẾNG VIỆT   │
│ 1. Làm sạch: HTML, URL, kéo dài ký tự, chuẩn hóa NFC  │
│ 2. Chuẩn hóa ngữ nghĩa: Teencode, Emoji token         │
│ 3. Lọc rác: Spam săn xu, mẫu mặc định, trùng lặp MD5  │
│ 4. Phân đoạn từ: Underthesea Word Segmentation        │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: CƠ SỞ DỮ LIỆU QUAN HỆ (DATABASE)         │
│ - Lưu dữ liệu sạch vào bảng SQLite/PostgreSQL          │
│ - Lưu trữ thông tin Seller, Product, Review sạch      │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: TRÍCH XUẤT ĐẶC TRƯNG & HUẤN LUYỆN AI      │
│ - Text Vectorization: TF-IDF (Word/Char) / PhoBERT     │
│ - Model 1: Sentiment Classifier (Pos / Neg / Neu)      │
│ - Model 2: ABSA Classifier (7 Aspects + Sentiments)   │
│ - Lưu weights mô hình vào thư mục models/ (.joblib/.pt)│
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 5: ĐÁNH GIÁ & KIỂM THỬ THỰC NGHIỆM          │
│ - Chạy trên tập Test độc lập                           │
│ - Đánh giá Accuracy, Precision, Recall, Macro-F1       │
│ - Vẽ Confusion Matrix & Báo cáo so sánh mô hình        │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 6: TRỰC QUAN HÓA & KHAI THÁC (STREAMLIT)    │
│ - Tab 1: Tổng quan & Quản trị Sản phẩm                 │
│ - Tab 2: Phân tích Cảm xúc theo Thời gian             │
│ - Tab 3: Phân tích Chi tiết 7 Khía cạnh (ABSA)        │
│ - Tab 4: Đám mây từ khóa & Báo cáo Nhà bán hàng       │
│ - Tab 5: Thử nghiệm Dự đoán Review Mới (Live Demo)    │
└────────────────────────────────────────────────────────┘
```

---

## 5. CẨM NANG TOÀN TẬP LÝ THUYẾT BẮT BUỘC PHẢI NẮM VỮNG

Khi bảo vệ đồ án trước hội đồng chấm thi, giảng viên sẽ tập trung hỏi sâu về bản chất học thuật, lý do lựa chọn công nghệ và các công thức toán học. Dưới đây là hệ thống lý thuyết cốt lõi được hệ thống hóa chi tiết:

---

### 5.1. Lý thuyết Xử lý Ngôn ngữ Tự nhiên tiếng Việt (Vietnamese NLP)

#### a) Chuẩn hóa bảng mã ký tự: Unicode dựng sẵn (NFC) vs Unicode tổ hợp (NFD)
- **Bản chất vấn đề:** Tiếng Việt có các chữ cái mang thanh điệu và dấu phụ (ví dụ: `à, ệ, ơ, ư`). Trong tin học tồn tại 2 cách mã hóa:
  - **Unicode Dựng sẵn (NFC - Normalization Form C):** Ký tự và dấu được gộp thành 1 code point duy nhất. Ví dụ: chữ `ế` là mã `\u1EBF` (chiếm 1 ký tự).
  - **Unicode Tổ hợp (NFD - Normalization Form D):** Ký tự gốc và dấu được tách rời. Ví dụ: chữ `ế` gồm chữ `e` + dấu mũ `^` + dấu sắc `'` (chiếm 3 code points: `\u0065\u0302\u0301`).
- **Tác động đến mô hình AI:** Nếu người dùng gõ bằng các bộ gõ Unikey khác nhau (một người gõ dựng sẵn, một người gõ tổ hợp), mô hình máy học sẽ xem hai từ đó là **hoàn toàn khác nhau**, dẫn đến bùng nổ kích thước từ điển và giảm độ chính xác.
- **Giải pháp:** Sử dụng thư viện `unicodedata.normalize('NFC', text)` để quy chuẩn 100% văn bản về dạng dựng sẵn.

#### b) Phân đoạn từ tiếng Việt (Vietnamese Word Segmentation)
- **Đặc trưng ngôn ngữ học:** Tiếng Anh là ngôn ngữ biến hình, ranh giới từ trùng với dấu cách (khoảng trắng). Tiếng Việt là ngôn ngữ đơn lập, âm tiết tính: dấu cách chỉ phân tách các âm tiết (syllables), không phân tách từ (words).
  - Ví dụ: *"giao hàng nhanh"* có 3 âm tiết, nhưng về mặt từ vựng gồm: `giao_hàng` (từ ghép mang nghĩa dịch vụ vận chuyển) và `nhanh` (tính từ). Nếu không tách từ mà để dấu cách mặc định, mô hình sẽ hiểu từ `giao` (giao nộp, giao lưu) và từ `hàng` (hàng hóa, xếp hàng) riêng lẻ, làm mất toàn bộ ngữ nghĩa.
- **Các kỹ thuật phân đoạn từ:**
  - *Phương pháp so khớp từ điển (Maximum Matching):* Dò từ dài nhất trong từ điển có sẵn (nhược điểm: không nhận biết được từ mới, từ đồng âm khác nghĩa).
  - *Phương pháp học máy thống kê (Conditional Random Fields - CRF):* Xem bài toán tách từ như bài toán gán nhãn chuỗi (chuỗi nhãn B - Beginning, I - Inside, E - End).
  - *Thư viện triển khai trong đồ án:* Sử dụng **`underthesea`** (sử dụng mô hình CRF kết hợp đặc trưng ngôn ngữ, đạt độ chính xác F1 > 97% trên tập dữ liệu tiếng Việt chuẩn).

#### c) Xử lý tiếng lóng, viết tắt (Teencode) và Biểu tượng cảm xúc (Emoji)
- **Teencode trong TMĐT:** Khách hàng đánh giá thường viết rất vắn tắt: `k` / `ko` (không), `sp` (sản phẩm), `dt` (điện thoại), `ship` (giao hàng), `re` (rẻ). Cần sử dụng bảng tra cứu từ điển (Dictionary Mapping) dựa trên cấu trúc `Hash Map / JSON` để ánh xạ về dạng chuẩn trước khi đưa vào mô hình.
- **Emoji Tokenization:** Biểu tượng cảm xúc mang hàm lượng thông tin cảm xúc cực lớn (đôi khi khách không viết chữ nào mà chỉ để `😍👍` hoặc `😡👎`).
  - Nếu xóa bỏ emoji: mất hoàn toàn tín hiệu cảm xúc.
  - Nếu để nguyên: emoji là các ký tự Unicode hiếm, làm tăng số chiều ma trận.
  - **Giải pháp tối ưu:** Ánh xạ emoji thành các token phân loại:
    - Nhóm tích cực: `❤️, 😍, 👍, ⭐` ➔ `_pos_emoji_`
    - Nhóm tiêu cực: `😡, 👎, 💔, 🤮` ➔ `_neg_emoji_`

#### d) Lọc dữ liệu rác (Spam) và Review trùng lặp (Duplicate Detection)
- **Spam trên sàn TMĐT:** Thường có 3 dạng chính:
  1. *Review săn xu (Coin Farming):* Khách copy lời bài hát, đoạn văn ngẫu nhiên hoặc gõ phím vô nghĩa (`asdfghjk`, `123456`) để đủ độ dài nhận xu.
  2. *Mẫu câu mặc định của sàn:* Khách bấm vào các nút gợi ý nhanh (`Chất lượng sản phẩm tuyệt vời`, `Thời gian giao hàng rất nhanh`) mà không tự viết.
  3. *Review quá ngắn:* Review dưới 2 từ hoặc chỉ chứa dấu chấm/dấu cách.
- **Thuật toán khử trùng lặp (BR12):** Sử dụng thuật toán băm **MD5 Hash** trên chuỗi văn bản đã chuẩn hóa. Nếu 2 review của cùng 1 sản phẩm có mã hash giống nhau 100%, review thứ hai sẽ bị gắn cờ `is_duplicate = True` và loại khỏi tập phân tích NLP.

---

### 5.2. Lý thuyết Biểu diễn Văn bản (Text Representation & Feature Engineering)

Mô hình học máy không thể đọc trực tiếp chuỗi ký tự, văn bản bắt buộc phải được chuyển đổi thành các vector số học (Vectorization).

#### a) Bag-of-Words (BoW)
- Mô hình "Túi đựng từ": Đếm số lần xuất hiện của từng từ trong câu mà không quan tâm đến thứ tự ngữ pháp.
- Hạn chế: Bị hiện tượng thưa thớt (Sparsity) và những từ xuất hiện quá nhiều nhưng vô nghĩa (như `và`, `là`, `của`) lại có trọng số cao nhất.

#### b) TF-IDF (Term Frequency - Inverse Document Frequency)
Đây là phương pháp nền tảng bắt buộc phải nắm vững công thức và ý nghĩa:

1. **Tần suất xuất hiện của từ trong tài liệu ($TF$ - Term Frequency):**
   $$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
   *(Trong đó $f_{t, d}$ là số lần từ $t$ xuất hiện trong văn bản $d$. $TF$ đo lường mức độ quan trọng cục bộ của từ trong bài viết).*

2. **Nghịch đảo tần suất văn bản ($IDF$ - Inverse Document Frequency):**
   $$IDF(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}| + 1}\right)$$
   *(Trong đó $N$ là tổng số review trong toàn bộ tập dữ liệu, mẫu số là số lượng review có chứa từ $t$. Mẫu số cộng thêm 1 để tránh lỗi chia cho 0).*

3. **Trọng số TF-IDF cuối cùng:**
   $$\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)$$

- **Ý nghĩa cốt lõi:**
  - Từ nào xuất hiện **nhiều lần trong 1 review cụ thể** (TF cao) nhưng **rất hiếm khi xuất hiện ở các review khác** (IDF cao) ➔ Được gán trọng số rất lớn vì đây là từ mang tính phân biệt mạnh (ví dụ: `hỏng_ngay`, `nứt_màn_hình`, `quá_rẻ`).
  - Từ nào xuất hiện phổ biến ở khắp mọi review (như `sản_phẩm`, `mua`, `shop`) ➔ Mẫu số của IDF xấp xỉ $N$, $\log(1) = 0$, trọng số bị ép về 0.

4. **Kỹ thuật N-grams kết hợp:**
   - **Word N-grams (1-2 words):** Lấy cả từ đơn (Unigram: `tốt`, `xấu`) và cặp từ liền kề (Bigram: `rất_tốt`, `không_thích`, `quá_chậm`). Bigram giải quyết được bài toán phủ định cơ bản.
   - **Char N-grams (3-5 characters):** Trích xuất các cụm ký tự liền kề để nhận diện từ ngay cả khi khách hàng gõ sai chính tả nhẹ (ví dụ: `đóng_goi`, `giao_han`).

#### c) Contextual Embeddings (Biểu diễn ngữ cảnh - PhoBERT / Transformer)
- Khác với TF-IDF hay Word2Vec (gán 1 vector cố định cho 1 từ bất kể ngữ cảnh), kiến trúc **Transformer (Self-Attention)** tính toán trọng số tương tác giữa tất cả các từ trong câu.
  - Ví dụ từ `hay`: trong câu *"Sách này rất **hay**"* (tích cực) mang nghĩa khác hoàn toàn với *"Sản phẩm này **hay** bị lỗi"* (tiêu cực).
  - PhoBERT sử dụng kiến trúc RoBERTa được huấn luyện sẵn trên 20GB văn bản tiếng Việt giúp vector của từ thay đổi linh hoạt theo từng ngữ cảnh câu.

---

### 5.3. Lý thuyết Các Thuật toán Học máy & Học sâu (Machine Learning & Deep Learning)

#### a) Mô hình Naïve Bayes (Multinomial Naïve Bayes)
- **Nguyên lý:** Dựa trên Định lý xác suất Bayes để tính xác suất một văn bản $d$ thuộc về lớp cảm xúc $c \in \{\text{Tích cực}, \text{Tiêu cực}, \text{Trung tính}\}$:
  $$P(c|d) = \frac{P(c) \cdot P(d|c)}{P(d)} \propto P(c) \prod_{i=1}^{n} P(w_i|c)$$
- **Giả định "Ngây thơ" (Naïve Assumption):** Giả định rằng sự xuất hiện của từ $w_i$ hoàn toàn độc lập với từ $w_j$ khi đã biết nhãn $c$. Mặc dù giả định này không đúng trong ngữ pháp tự nhiên, Naïve Bayes vẫn hoạt động cực kỳ hiệu quả và nhanh trên bài toán phân loại văn bản.
- **Kỹ thuật làm mịn Laplace (Laplace Smoothing):**
  $$P(w|c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w'} \text{count}(w', c) + \alpha \cdot |V|}$$
  *(Với $\alpha = 1$, giúp tránh trường hợp khi gặp 1 từ mới chưa từng xuất hiện trong tập Train làm xác suất $P(w|c) = 0$, kéo theo toàn bộ tích xác suất bằng 0).*

#### b) Mô hình Support Vector Machine (SVM)
- **Nguyên lý:** Tìm một siêu phẳng (Hyperplane) trong không gian nhiều chiều để phân tách các lớp dữ liệu sao cho **Khoảng cách biên (Margin) giữa siêu phẳng và các điểm dữ liệu gần nhất (Support Vectors) là lớn nhất có thể**.
- **Phương trình siêu phẳng:**
  $$w^T x + b = 0$$
- **Hàm mục tiêu tối ưu lồi (Soft-margin SVM):**
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i$$
  *(Trong đó $C$ là siêu tham số kiểm soát sự đánh đổi giữa việc mở rộng margin và việc chấp nhận một số điểm bị phân loại sai; $\xi_i$ là biến bù trừ lỗi).*
- **Kernel Trick:** Khi dữ liệu không thể phân tách tuyến tính trong không gian ban đầu, SVM sử dụng các hàm nhân (như RBF, Polynomial) để ánh xạ phi tuyến dữ liệu lên không gian số chiều cao hơn, nơi dữ liệu trở nên phân tách tuyến tính.
- **Ưu điểm trong NLP:** Phân loại văn bản có số chiều đặc trưng rất lớn (hàng nghìn từ trong TF-IDF). SVM cực kỳ mạnh mẽ trong không gian số chiều cao và ít bị hiện tượng Overfitting hơn các mô hình khác.

#### c) Mô hình Bi-LSTM (Bidirectional Long Short-Term Memory)
- **Mạng nơ-ron hồi quy (RNN):** Phù hợp với dữ liệu chuỗi thời gian/văn bản. Tuy nhiên RNN truyền thống bị vấn đề tiêu biến đạo hàm (Vanishing Gradient) khi câu quá dài.
- **Cơ chế hoạt động của LSTM Cell:** Sử dụng 3 cổng (Gates) và 1 trạng thái ô nhớ (Cell State $C_t$):
  1. *Cổng quên (Forget Gate $f_t$):* Quyết định bỏ đi thông tin nào từ quá khứ: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$.
  2. *Cổng vào (Input Gate $i_t$):* Quyết định cập nhật thông tin mới nào vào ô nhớ.
  3. *Cổng ra (Output Gate $o_t$):* Quyết định trạng thái ẩn tiếp theo $h_t$.
- **Bi-directional (Hai chiều):** Mạng gồm 2 nhánh LSTM song song: 1 nhánh đọc câu từ trái qua phải, 1 nhánh đọc từ phải qua trái. Điều này giúp từ hiện tại nắm bắt được cả ngữ cảnh đứng trước lẫn đứng sau nó.

#### d) Mô hình Transformer & PhoBERT
- **Kiến trúc:** PhoBERT sử dụng cấu trúc **Transformer Encoder**. Cơ chế cốt lõi là **Multi-Head Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  *(Trong đó $Q$ là Query, $K$ là Key, $V$ là Value. Cơ chế này tính toán ma trận tương đồng giữa mọi cặp từ trong câu, cho phép mô hình ghi nhận sự phụ thuộc xa giữa các từ).*
- **Quy trình Fine-tuning:**
  1. Nạp mô hình PhoBERT đã được Pretrain trên kho ngữ liệu tiếng Việt khổng lồ.
  2. Thêm một lớp phân loại tuyến tính (Classification Head / Fully Connected Layer) ở đầu ra của token đặc biệt `[CLS]`.
  3. Huấn luyện toàn bộ mạng với tốc độ học nhỏ (Learning Rate nhỏ, e.g., $2 \times 10^{-5}$) trên tập dữ liệu đánh giá sản phẩm của nhóm.

---

### 5.4. Lý thuyết Phân tích Cảm xúc theo Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)

ABSA là bài toán nâng cao chia nhỏ cảm xúc tổng thể thành các khía cạnh chi tiết.

#### a) Định nghĩa bài toán
Một câu đánh giá $R$ có thể đề cập đến một tập hợp con các khía cạnh $A \subseteq \{A_1, A_2, \dots, A_7\}$ với:
1. $A_1$: Chất lượng sản phẩm (`Quality`)
2. $A_2$: Giá cả (`Price`)
3. $A_3$: Thiết kế / Kiểu dáng (`Design`)
4. $A_4$: Độ bền / Thời lượng pin (`Durability`)
5. $A_5$: Đóng gói (`Packaging`)
6. $A_6$: Vận chuyển / Giao hàng (`Delivery`)
7. $A_7$: Dịch vụ chăm sóc khách hàng (`Service`)

Mỗi khía cạnh $A_k$ xuất hiện sẽ mang một giá trị cảm xúc $S_k \in \{\text{Tích cực}, \text{Tiêu cực}, \text{Trung tính}\}$.

#### b) Phân biệt Multi-class và Multi-label
- **Multi-class (Đa lớp):** Mỗi mẫu dữ liệu chỉ thuộc về **duy nhất 1 lớp** (Ví dụ: Cảm xúc tổng thể chỉ có thể là 1 trong 3 nhãn: Pos, Neg hoặc Neu). Dùng hàm kích hoạt **Softmax** ở đầu ra: $\sum P_i = 1$.
- **Multi-label (Đa nhãn):** Mỗi mẫu dữ liệu có thể thuộc về **nhiều lớp cùng một lúc** (Ví dụ: Một câu có thể vừa nói về `Delivery`, vừa nói về `Quality` và `Price`).
  - Dùng hàm kích hoạt **Sigmoid** độc lập cho từng đầu ra của khía cạnh: $P(A_k) = \frac{1}{1 + e^{-z_k}}$.
  - Khía cạnh được coi là xuất hiện nếu $P(A_k) \ge \text{Threshold}$ (thường chọn ngưỡng 0.5).

---

### 5.5. Lý thuyết Đánh giá & Kiểm thử Mô hình (Evaluation Metrics)

Khi dữ liệu thực tế trên sàn TMĐT bị lệch (review 5 sao chiếm đa số ~70%, review 1-2 sao chiếm ~15%), việc nắm vững các chỉ số đánh giá là yếu tố then chốt để hội đồng đánh giá cao tư duy học thuật của nhóm.

#### a) Ma trận nhầm lẫn (Confusion Matrix)
Ma trận thể hiện chi tiết số lượng dự đoán đúng và sai:

| Thực tế \ Dự đoán | Dự đoán là Dương tính (Positive) | Dự đoán là Âm tính (Negative) |
| :--- | :---: | :---: |
| **Thực tế Dương tính** | **TP** *(True Positive - Đoán đúng Pos)* | **FN** *(False Negative - Bị bỏ sót)* |
| **Thực tế Âm tính** | **FP** *(False Positive - Báo động giả)* | **TN** *(True Negative - Đoán đúng Neg)* |

#### b) Các chỉ số đo lường chuẩn
1. **Accuracy (Độ chính xác toàn cục):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   *(Cạm bẫy: Nếu tập dữ liệu có 90 câu Tích cực và 10 câu Tiêu cực, một mô hình ngớ ngẩn luôn luôn đoán "Tích cực" sẽ đạt Accuracy = 90%, nhưng hoàn toàn vô dụng vì không phát hiện được bất kỳ câu Tiêu cực nào).*

2. **Precision (Độ chuẩn xác / Độ chính xác của dự đoán):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   *(Trả lời câu hỏi: Trong tất cả các câu mà máy dự đoán là Tiêu cực, có bao nhiêu câu thực sự là Tiêu cực?).*

3. **Recall (Độ phủ / Độ nhạy):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   *(Trả lời câu hỏi: Trong tất cả các câu Tiêu cực thực tế ngoài đời, máy đã phát hiện và tóm được bao nhiêu câu?).*

4. **F1-Score (Điểm F1):**
   Trung bình điều hòa giữa Precision và Recall. Nếu 1 trong 2 chỉ số quá thấp, F1-score sẽ lập tức bị kéo xuống:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### c) Phân biệt Macro-F1, Micro-F1 và Weighted-F1
Trong bài toán phân loại 3 lớp (Pos, Neg, Neu) hoặc đa nhãn:
- **Macro-F1 (Thước đo vàng cho đồ án này):** Tính F1 riêng cho từng lớp rồi lấy trung bình cộng đều:
  $$\text{Macro-F1} = \frac{F_{1,\text{Pos}} + F_{1,\text{Neg}} + F_{1,\text{Neu}}}{3}$$
  *Ưu điểm:* Đánh giá bình đẳng mọi lớp. Nếu mô hình đoán kém ở lớp Tiêu cực (lớp thiểu số), Macro-F1 sẽ tụt dốc ngay lập tức, phản ánh đúng thực chất mô hình.
- **Micro-F1:** Gộp toàn bộ TP, FP, FN của tất cả các lớp lại rồi mới tính. Micro-F1 bị chi phối nặng nề bởi lớp chiếm đa số.
- **Weighted-F1:** Tính trung bình có trọng số theo tỷ lệ số lượng mẫu của từng lớp.

---

### 5.6. Lý thuyết Thiết kế Cơ sở Dữ liệu Quan hệ (RDBMS)

#### a) Các dạng chuẩn hóa CSDL (Database Normalization)
- **Chuẩn 1 (1NF):** Mỗi cột chỉ chứa giá trị nguyên tố (nguyên tử - atomic), không chứa danh sách hoặc mảng lặp lại.
- **Chuẩn 2 (2NF):** Đạt 1NF và mọi thuộc tính không khóa phải phụ thuộc hàm hoàn toàn vào khóa chính (không phụ thuộc vào một phần của khóa ghép).
- **Chuẩn 3 (3NF):** Đạt 2NF và không có thuộc tính không khóa nào phụ thuộc bắc cầu vào khóa chính ($A \to B$ và $B \to C$).

#### b) Thiết kế Lược đồ Bảng Thực tế cho Đồ án:
Nhóm cần triển khai lược đồ quan hệ gồm 5 bảng chuẩn hóa sau:

```sql
-- 1. Bảng Nhà bán hàng (Seller)
CREATE TABLE sellers (
    seller_id VARCHAR(50) PRIMARY KEY,
    seller_name VARCHAR(255) NOT NULL,
    platform VARCHAR(50) NOT NULL  -- 'tiki', 'shopee'
);

-- 2. Bảng Sản phẩm (Products)
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    seller_id VARCHAR(50) REFERENCES sellers(seller_id),
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    platform VARCHAR(50) NOT NULL
);

-- 3. Bảng Bình luận gốc và tiền xử lý (Reviews)
CREATE TABLE reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50) REFERENCES products(product_id),
    rating INT CHECK (rating BETWEEN 1 AND 5),
    comment_text_raw TEXT,
    comment_text_clean TEXT,
    comment_text_segmented TEXT,
    created_at TIMESTAMP,
    like_count INT DEFAULT 0,
    is_spam BOOLEAN DEFAULT FALSE,
    is_duplicate BOOLEAN DEFAULT FALSE,
    is_valid_for_analysis BOOLEAN DEFAULT TRUE
);

-- 4. Bảng Kết quả Dự đoán Cảm xúc Tổng thể (Sentiment Predictions)
CREATE TABLE sentiment_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(50) REFERENCES reviews(review_id),
    predicted_sentiment VARCHAR(20), -- 'Positive', 'Negative', 'Neutral'
    confidence_score FLOAT,
    model_version VARCHAR(50)
);

-- 5. Bảng Kết quả Dự đoán theo Khía cạnh (Aspect Predictions)
CREATE TABLE aspect_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(50) REFERENCES reviews(review_id),
    aspect_name VARCHAR(50), -- 'delivery', 'packaging', 'quality', 'price', etc.
    aspect_sentiment VARCHAR(20), -- 'Positive', 'Negative', 'Neutral'
    confidence_score FLOAT
);
```

---

## 6. BỘ CÂU HỎI VẤN ĐÁP THƯỜNG GẶP KHI BẢO VỆ ĐỒ ÁN (VÀ GỢI Ý TRẢ LỜI)

Dưới đây là các câu hỏi mà hội đồng chấm thi thường xuyên hỏi sinh viên, nhóm nên học thuộc cách giải thích:

### Câu hỏi 1: Tại sao nhóm không dùng luôn số sao rating (1-5 sao) làm nhãn cảm xúc mà phải tốn công xây dựng mô hình NLP?
* **Gợi ý trả lời:**  
  *"Thưa thầy/cô, việc sử dụng trực tiếp số sao rating có 2 hạn chế rất lớn trong thực tế:*  
  *Thứ nhất, tồn tại hiện tượng **Nghịch lý Rating - Nội dung** (Rating-Text Contradiction): Nhiều khách hàng bấm nhầm 5 sao nhưng nội dung bên dưới lại mắng shop thậm tệ, hoặc ngược lại khách khen sản phẩm nhưng bấm nhầm 1 sao. Ngoài ra, nhiều khách có thói quen đánh giá châm biếm/mỉa mai (ví dụ: chấm 5 sao kèm câu 'Giao hàng nhanh quá, đợi có 2 tháng là tới'). Nếu chỉ nhìn số sao, hệ thống sẽ bị đánh lừa hoàn toàn.*  
  *Thứ hai, số sao chỉ thể hiện cảm xúc chung chung cho cả đơn hàng, **hoàn toàn không thể bóc tách được các khía cạnh cụ thể (ABSA)**. Khách hàng chấm 4 sao nhưng trong câu văn họ khen máy đẹp mà chê shipper thô lỗ. Chỉ có mô hình NLP mới có thể bóc tách được chính xác từng khía cạnh để báo cáo cho người bán."*

### Câu hỏi 2: Dữ liệu bình luận trên sàn thương mại điện tử bị mất cân bằng trầm trọng (5 sao chiếm phần lớn, 1-2 sao rất ít). Nhóm đã xử lý vấn đề này như thế nào?
* **Gợi ý trả lời:**  
  *"Thưa thầy/cô, nhóm nhận diện rõ hiện tượng Imbalanced Data ngay từ bước EDA. Để giải quyết, nhóm đã áp dụng đồng thời các giải pháp:*  
  *1. Về mặt dữ liệu: Thu thập thêm các sản phẩm có nhiều tranh cãi để tăng mẫu tiêu cực; sử dụng kỹ thuật Undersampling lớp đa số và Oversampling / Text Augmentation (thay thế từ đồng nghĩa) cho lớp thiểu số.*  
  *2. Về mặt thuật toán: Sử dụng tham số `class_weight='balanced'` trong mô hình SVM / Logistic Regression để phạt nặng hơn khi mô hình đoán sai lớp thiểu số.*  
  *3. Về mặt đánh giá: Nhóm tuyệt đối không dùng Accuracy làm thước đo chính mà sử dụng **Macro-F1 score** để đánh giá công bằng hiệu năng trên cả 3 lớp cảm xúc."*

### Câu hỏi 3: Sự khác biệt bản chất giữa PhoBERT và các mô hình học máy truyền thống (SVM, Naïve Bayes) là gì?
* **Gợi ý trả lời:**  
  *"Thưa thầy/cô, sự khác biệt căn bản nằm ở **Khả năng nắm bắt ngữ cảnh (Contextual Understanding)**:*  
  *Mô hình truyền thống (SVM, Naïve Bayes) kết hợp TF-IDF coi văn bản là một 'túi từ', mỗi từ được gán trọng số độc lập và mất đi thứ tự ngữ pháp. Mô hình không thể phân biệt được từ đồng âm khác nghĩa hoặc các cấu trúc câu phức tạp.*  
  *Ngược lại, PhoBERT dựa trên kiến trúc Transformer với cơ chế Self-Attention hai chiều. Vector nhúng của từ trong PhoBERT thay đổi linh hoạt theo toàn bộ các từ xung quanh nó trong câu. Do đó PhoBERT hiểu được các cấu trúc phủ định phức tạp, từ lóng và sắc thái mỉa mai tốt hơn rất nhiều so với mô hình truyền thống."*

### Câu hỏi 4: Nhóm đã giải quyết hiện tượng Teencode và lỗi chính tả tiếng Việt như thế nào?
* **Gợi ý trả lời:**  
  *"Thưa thầy/cô, nhóm đã xây dựng một Pipeline tiền xử lý đa tầng chuyên biệt cho TMĐT:*  
  *Đầu tiên, chuẩn hóa bảng mã về Unicode dựng sẵn (NFC). Sau đó, sử dụng từ điển quy chuẩn (`teencode_dict.json`) được thu thập từ thực tế mua sắm trực tuyến để chuyển đổi các từ viết tắt về từ chuẩn tiếng Việt. Đối với các biểu tượng cảm xúc, nhóm ánh xạ thành các token đại diện ngữ nghĩa như `_pos_emoji_`, `_neg_emoji_`. Cuối cùng, văn bản sạch mới được đưa qua công cụ `underthesea` để tách từ chuẩn xác."*

---

**CHÚC HAI BẠN LONG VÀ MẠNH ÔN TẬP TỐT VÀ HOÀN THÀNH XUẤT SẮC ĐỒ ÁN TỐT NGHIỆP!**

