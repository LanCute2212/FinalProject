import argparse
import sys
import os
import glob
from pathlib import Path
import pandas as pd

# Thêm đường dẫn gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.logger import get_logger

logger = get_logger("run_preprocess")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_input_directory(raw_dir: str) -> Path:
    directory = Path(raw_dir)
    if not directory.is_absolute() and not directory.exists():
        rooted = PROJECT_ROOT / directory
        if rooted.exists():
            directory = rooted
    return directory

def load_all_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Tự động quét và gộp tất cả các file raw CSV trong thư mục data/raw/
    (Bao gồm Tiki, Sample, Shopee và bất kỳ sàn nào cào sau này).
    """
    raw_path = _resolve_input_directory(raw_dir)
    csv_files = sorted(glob.glob(str(raw_path / "*_raw.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(str(raw_path / "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy bất kỳ file CSV nào trong '{raw_path}'!")

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
    
    # Review ID chỉ duy nhất trong phạm vi một nền tảng. Không gộp các review
    # khác sàn, và không biến ID thiếu thành chuỗi "nan" rồi xóa nhầm các dòng.
    if "review_id" in combined_df.columns:
        review_ids = combined_df["review_id"].astype("string").str.strip()
        missing_ids = review_ids.isna() | review_ids.str.lower().isin({"", "nan", "none", "null", "n/a"})
        review_ids = review_ids.mask(missing_ids, pd.NA)
        combined_df["review_id"] = review_ids
        duplicate_keys = pd.DataFrame(index=combined_df.index)
        if "platform" in combined_df.columns:
            duplicate_keys["platform"] = (
                combined_df["platform"].astype("string").fillna("").str.strip().str.lower()
            )
        duplicate_keys["review_id"] = review_ids.fillna("")
        duplicate_mask = duplicate_keys.duplicated(keep="last") & ~missing_ids
        dedup_count = int(duplicate_mask.sum())
        combined_df = combined_df.loc[~duplicate_mask].reset_index(drop=True)
        if dedup_count > 0:
            logger.info(f"Đã loại bỏ {dedup_count} bản ghi trùng ID trong cùng nền tảng.")

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
        originally_missing = (
            combined_df["comment_text"].isna()
            | combined_df["comment_text"].astype(str).str.strip().eq("")
            | combined_df["comment_text"].astype(str).str.strip().str.lower().eq("nan")
        )
        recovered_count = (originally_missing & recovered.notna() & recovered.astype(str).str.strip().ne("")).sum()
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
    low_information = df["low_information"].sum() if "low_information" in df else 0
    percentage = lambda count, denominator=total: (count / denominator * 100) if denominator else 0.0

    print("\n" + "="*70)
    print("           BÁO CÁO THỐNG KÊ TIỀN XỬ LÝ DỮ LIỆU (DATA REPORT)")
    print("="*70)
    print(f" Tổng số review thô (Raw Records)       : {total:>6}")
    print(f" Review hợp lệ cho NLP (Valid Records)   : {valid:>6} ({percentage(valid):>5.1f}%)")
    print(f" Review rác / Săn xu (Spam Detected)     : {spam:>6} ({percentage(spam):>5.1f}%)")
    print(f" Review trùng lặp nội dung (Duplicates)  : {duplicate:>6} ({percentage(duplicate):>5.1f}%)")
    print(f" Review ít thông tin (Không xem là spam)  : {low_information:>6} ({percentage(low_information):>5.1f}%)")
    print("-" * 70)

    # Phân bố nền tảng
    if "platform" in df.columns:
        print(" Phân bố theo Nền tảng (Platform breakdown):")
        p_counts = df["platform"].value_counts()
        for p_name, p_cnt in p_counts.items():
            print(f"   • {p_name:<20}: {p_cnt:>5} reviews ({percentage(p_cnt):>5.1f}%)")
        print("-" * 70)

    # Phân bố danh mục sản phẩm
    if "category" in df.columns:
        print(" Phân bố theo Danh mục sản phẩm (Category breakdown):")
        c_counts = df["category"].value_counts()
        for c_name, c_cnt in c_counts.items():
            print(f"   • {c_name:<30}: {c_cnt:>5} reviews ({percentage(c_cnt):>5.1f}%)")
        print("-" * 70)

    # Phân bố rating của tập hợp lệ
    valid_df = df[df["is_valid_for_analysis"]]
    if not valid_df.empty:
        print(" Phân bố Rating của tập dữ liệu hợp lệ (1 - 5 sao):")
        if "rating_is_valid" in valid_df.columns:
            rating_df = valid_df[valid_df["rating_is_valid"]]
        else:
            rating_df = valid_df
        rating_counts = rating_df["rating"].value_counts().sort_index()
        for r, cnt in rating_counts.items():
            rating_total = len(rating_df) or 1
            bar = "█" * int(cnt / rating_total * 30)
            print(f"   ★ {r} sao : {cnt:>4} reviews ({cnt/rating_total*100:>5.1f}%) | {bar}")

        print("-" * 70)
        print(" Phân tích độ dài văn bản (Số lượng từ / Review hợp lệ):")
        print(f"   - Trung bình (Mean words) : {valid_df['word_count'].mean():.1f} từ")
        print(f"   - Trung vị   (Median words): {valid_df['word_count'].median():.1f} từ")
        print(f"   - Tối đa     (Max words)   : {valid_df['word_count'].max():.1f} từ")
        print(f"   - Tối thiểu  (Min words)   : {valid_df['word_count'].min():.1f} từ")

    print("="*70)
    print("\n[MẪU VĂN BẢN TRƯỚC VÀ SAU KHI TIỀN XỬ LÝ (AUDIT MẪU)]:")
    samples = df.sample(min(5, len(df)), random_state=42) if len(df) else df
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
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
