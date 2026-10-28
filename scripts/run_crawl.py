import argparse
import sys
import os

# Thêm đường dẫn gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawler.tiki_crawler import TikiCrawler
from src.crawler.shopee_crawler import ShopeeCrawler
from src.utils.logger import get_logger

logger = get_logger("run_crawl")

def main():
    parser = argparse.ArgumentParser(description="Công cụ cào dữ liệu đánh giá sàn TMĐT (Shopee / Tiki)")
    parser.add_argument("--platform", type=str, choices=["tiki", "shopee"], default="tiki", help="Nền tảng TMĐT cần cào")
    parser.add_argument("--product_id", type=str, required=True, help="Mã sản phẩm (product_id hoặc item_id)")
    parser.add_argument("--shop_id", type=str, default="auto", help="Mã cửa hàng (shop_id), mặc định auto")
    parser.add_argument("--product_name", type=str, default=None, help="Tên sản phẩm (tự động lấy nếu bỏ trống)")
    parser.add_argument("--max_reviews", type=int, default=100, help="Số lượng review tối đa cần cào (nhập 0 hoặc dùng --all để cào hết)")
    parser.add_argument("--all", action="store_true", help="Cào TOÀN BỘ review của sản phẩm không giới hạn số lượng")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Thư mục lưu trữ raw data")
    parser.add_argument("--no_preprocess", action="store_true", help="Không tự động chạy tiền xử lý sau khi cào")

    args = parser.parse_args()

    max_reviews = 0 if args.all else args.max_reviews

    logger.info(f"Khởi động Crawler cho sàn: {args.platform.upper()} (Chế độ: {'Cào toàn bộ' if max_reviews == 0 else f'Tối đa {max_reviews}'})")

    if args.platform == "tiki":
        crawler = TikiCrawler()
        crawler.fetch_product_reviews(
            shop_id=args.shop_id,
            product_id=args.product_id,
            product_name=args.product_name,
            max_reviews=max_reviews
        )
    elif args.platform == "shopee":
        if args.shop_id == "auto":
            logger.error("Shopee API yêu cầu phải cung cấp --shop_id cụ thể của cửa hàng!")
            sys.exit(1)
        crawler = ShopeeCrawler()
        crawler.fetch_product_reviews(
            shop_id=args.shop_id,
            product_id=args.product_id,
            product_name=args.product_name,
            max_reviews=max_reviews
        )

    # Lưu và gom dữ liệu vào 1 file duy nhất cho nền tảng đó
    saved_files = crawler.save_raw_data(output_dir=args.output_dir)
    if saved_files:
        logger.info(f"[HOÀN TẤT CÀO DỮ LIỆU] Dữ liệu đã đồng bộ vào bộ '{args.platform}': {saved_files['csv']}")
        
        # Tự động cập nhật tập processed nếu không bật cờ --no_preprocess
        if not args.no_preprocess:
            logger.info("Tự động kích hoạt tiền xử lý dữ liệu để cập nhật Dashboard...")
            try:
                from scripts.run_preprocess import run_full_preprocessing
                run_full_preprocessing(raw_dir=args.output_dir)
            except Exception as e:
                logger.warning(f"Lỗi khi tự động chạy tiền xử lý: {e}. Bạn có thể chạy thủ công: python scripts/run_preprocess.py")
    else:
        logger.warning("[CẢNH BÁO] Không thu thập được dữ liệu review nào.")

if __name__ == "__main__":
    main()
