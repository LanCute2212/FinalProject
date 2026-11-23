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

def load_all_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Tự động quét và gộp tất cả các file raw CSV trong thư mục data/raw/
    (Bao gồm Tiki, Sample, Shopee và bất kỳ sàn nào cào sau này).
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*_raw.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy bất kỳ file CSV nào trong '{raw_dir}'!")

    dfs = []
    logger.info(f"Tìm thấy {len(csv_files)} file raw dữ liệu:")
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            logger.info(f" - Nạp '{os.path.basename(f)}': {len(temp_df)} bản ghi")
            dfs.append(temp_df)
        except Exception as e:
            logger.warning(f"Lỗi khi đọc file {f}: {e}")

    if not dfs:
        raise ValueError("Không có dữ liệu hợp lệ nào được nạp từ các file raw.")

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Khử trùng lặp theo review_id nếu có
    if "review_id" in combined_df.columns:
        combined_df["review_id"] = combined_df["review_id"].astype(str)
        before_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["review_id"], keep="last")
        dedup_count = before_len - len(combined_df)
        if dedup_count > 0:
            logger.info(f"Đã loại bỏ {dedup_count} bản ghi trùng lặp giữa các bộ.")

    # Tự động phục hồi comment_text từ raw_payload nếu cột comment_text bị rỗng/NaN
    if "raw_payload" in combined_df.columns and "comment_text" in combined_df.columns:
        import json
        def recover_text(row):
            txt = row.get("comment_text")
            if pd.isna(txt) or str(txt).strip() == "" or str(txt).strip().lower() == "nan":
                payload = row.get("raw_payload")
                if pd.notna(payload):
                    try:
                        data = json.loads(payload) if isinstance(payload, str) else payload
                        if isinstance(data, dict):
                            content = data.get("content")
                            if content and str(content).strip():
                                return str(content).strip()
                    except Exception:
                        pass
            return txt
        recovered = combined_df.apply(recover_text, axis=1)
        recovered_count = (combined_df["comment_text"].isna() & recovered.notna()).sum()
        if recovered_count > 0:
            logger.info(f"Đã tự động trích xuất nội dung từ raw_payload cho {recovered_count} review có comment_text bị trống.")
        combined_df["comment_text"] = recovered

    # Đảm bảo các cột metadata mới luôn đầy đủ và có kiểu dữ liệu chuẩn
    if "category" not in combined_df.columns:
        combined_df["category"] = "Chưa phân loại"
    else:
        combined_df["category"] = combined_df["category"].fillna("Chưa phân loại")

    if "reviewer_name" not in combined_df.columns:
        combined_df["reviewer_name"] = "Khách hàng"
    else:
        combined_df["reviewer_name"] = combined_df["reviewer_name"].fillna("Khách hàng")

    if "like_count" not in combined_df.columns:
        combined_df["like_count"] = 0
    else:
        combined_df["like_count"] = pd.to_numeric(combined_df["like_count"], errors="coerce").fillna(0).astype(int)

    logger.info(f"Tổng hợp toàn bộ dữ liệu raw: {len(combined_df)} bản ghi.")
    return combined_df

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

    # Phân bố nền tảng
    if "platform" in df.columns:
        print(" Phân bố theo Nền tảng (Platform breakdown):")
        p_counts = df["platform"].value_counts()
        for p_name, p_cnt in p_counts.items():
            print(f"   • {p_name:<20}: {p_cnt:>5} reviews ({p_cnt/total*100:>5.1f}%)")
        print("-" * 70)

    # Phân bố danh mục sản phẩm
    if "category" in df.columns:
        print(" Phân bố theo Danh mục sản phẩm (Category breakdown):")
        c_counts = df["category"].value_counts()
        for c_name, c_cnt in c_counts.items():
            print(f"   • {c_name:<30}: {c_cnt:>5} reviews ({c_cnt/total*100:>5.1f}%)")
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
        print(f"\nID: {row.get('review_id', idx)} | Sàn: {row.get('platform')} | SKU: {row.get('product_id')} | Rating: {row.get('rating')}★ | Trạng thái: [{status}]")
        print(f" • GỐC (RAW)       : {row['comment_text_raw']}")
        print(f" • CHUẨN HÓA (CLEAN): {row['comment_text_clean']}")
        print(f" • TÁCH TỪ (SEGMENT): {row['comment_text_segmented']}")
    print("\n" + "="*70 + "\n")

def run_full_preprocessing(raw_dir: str = "data/raw", input_file: str = None, output_path: str = "data/processed/reviews_processed.csv") -> pd.DataFrame:
    """Hàm chạy toàn diện tiền xử lý dữ liệu từ tất cả raw files hoặc file chỉ định."""
    pipeline = PreprocessingPipeline()
    
    if input_file:
        logger.info(f"Tiền xử lý file đơn lẻ: {input_file}")
        processed_df = pipeline.run_pipeline(input_path=input_file, output_path=output_path)
    else:
        logger.info(f"Tiền xử lý và gộp TẤT CẢ các bộ dữ liệu trong '{raw_dir}'...")
        combined_df = load_all_raw_data(raw_dir=raw_dir)
        processed_df = pipeline.process_dataframe(combined_df)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu kết quả gộp vào: {output_path}")

    display_statistics(processed_df)
    return processed_df

def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline Tiền xử lý dữ liệu Customer Voice Analytics")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn file raw CSV/JSON đầu vào (nếu bỏ trống sẽ gộp tất cả)")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Thư mục raw data")
    parser.add_argument("--output", type=str, default="data/processed/reviews_processed.csv", help="Đường dẫn lưu file sạch")

    args = parser.parse_args()
    run_full_preprocessing(raw_dir=args.raw_dir, input_file=args.input, output_path=args.output)

if __name__ == "__main__":
    main()
