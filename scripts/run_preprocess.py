import argparse
import sys
import os
import glob
import pandas as pd

# Thêm đường dẫn gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.logger import get_logger

logger = get_logger("run_preprocess")

def find_latest_raw_file(raw_dir: str = "data/raw") -> str:
    """Tự động tìm file raw mới nhất trong thư mục nếu người dùng không truyền tham số."""
    csv_files = glob.glob(os.path.join(raw_dir, "*_raw.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy bất kỳ file CSV nào trong thư mục '{raw_dir}'. Hãy cào dữ liệu hoặc chạy generate_sample_data.py trước!")
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def display_statistics(df: pd.DataFrame):
    """In bảng thống kê chi tiết định lượng cho báo cáo đồ án."""
    total = len(df)
    valid = df["is_valid_for_analysis"].sum()
    spam = df["is_spam"].sum()
    duplicate = df["is_duplicate"].sum()

    print("\n" + "="*70)
    print("           BÁO CÁO THỐNG KÊ TIỀN XỬ LÝ DỮ LIỆU (DATA REPORT)")
    print("="*70)
    print(f" Tổng số review thô (Raw Records)       : {total:>6}")
    print(f" Review hợp lệ cho NLP (Valid Records)   : {valid:>6} ({valid/total*100:>5.1f}%)")
    print(f" Review rác / Săn xu (Spam Detected)     : {spam:>6} ({spam/total*100:>5.1f}%)")
    print(f" Review trùng lặp nội dung (Duplicates)  : {duplicate:>6} ({duplicate/total*100:>5.1f}%)")
    print("-" * 70)

    # Phân bố rating của tập hợp lệ
    valid_df = df[df["is_valid_for_analysis"]]
    if not valid_df.empty:
        print(" Phân bố Rating của tập dữ liệu hợp lệ (1 - 5 sao):")
        rating_counts = valid_df["rating"].value_counts().sort_index()
        for r, cnt in rating_counts.items():
            bar = "█" * int(cnt / len(valid_df) * 30)
            print(f"   ★ {r} sao : {cnt:>4} reviews ({cnt/len(valid_df)*100:>5.1f}%) | {bar}")

        print("-" * 70)
        print(" Phân tích độ dài văn bản (Số lượng từ / Review hợp lệ):")
        print(f"   - Trung bình (Mean words) : {valid_df['word_count'].mean():.1f} từ")
        print(f"   - Trung vị   (Median words): {valid_df['word_count'].median():.1f} từ")
        print(f"   - Tối đa     (Max words)   : {valid_df['word_count'].max():.1f} từ")
        print(f"   - Tối thiểu  (Min words)   : {valid_df['word_count'].min():.1f} từ")

    print("="*70)
    print("\n[MẪU VĂN BẢN TRƯỚC VÀ SAU KHI TIỀN XỬ LÝ (AUDIT MẪU)]:")
    samples = df.sample(min(5, len(df)), random_state=42)
    for idx, row in samples.iterrows():
        status = row.get("filter_status", "VALID")
        print(f"\nID: {row.get('review_id', idx)} | SKU: {row.get('product_id')} | Rating: {row.get('rating')}★ | Trạng thái: [{status}]")
        print(f" • GỐC (RAW)       : {row['comment_text_raw']}")
        print(f" • CHUẨN HÓA (CLEAN): {row['comment_text_clean']}")
        print(f" • TÁCH TỪ (SEGMENT): {row['comment_text_segmented']}")
    print("\n" + "="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline Tiền xử lý dữ liệu Customer Voice Analytics")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn file raw CSV/JSON đầu vào")
    parser.add_argument("--output", type=str, default="data/processed/reviews_processed.csv", help="Đường dẫn lưu file sạch")

    args = parser.parse_args()

    input_path = args.input or find_latest_raw_file()
    logger.info(f"Sử dụng file dữ liệu đầu vào: {input_path}")

    pipeline = PreprocessingPipeline()
    processed_df = pipeline.run_pipeline(input_path=input_path, output_path=args.output)

    display_statistics(processed_df)

if __name__ == "__main__":
    main()
