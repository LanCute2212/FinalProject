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
    parser.add_argument("--product_name", type=str, default="Sản phẩm TMĐT", help="Tên sản phẩm")
    parser.add_argument("--max_reviews", type=int, default=100, help="Số lượng review tối đa cần cào")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Thư mục lưu trữ raw data")

    args = parser.parse_args()

    logger.info(f"Khởi động Crawler cho sàn: {args.platform.upper()}")

    if args.platform == "tiki":
        crawler = TikiCrawler()
        crawler.fetch_product_reviews(
            shop_id=args.shop_id,
            product_id=args.product_id,
            product_name=args.product_name,
            max_reviews=args.max_reviews
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
            max_reviews=args.max_reviews
        )

    saved_files = crawler.save_raw_data(output_dir=args.output_dir, filename_prefix=f"{args.platform}_{args.product_id}")
    if saved_files:
        logger.info(f"[HOÀN TẤT CÀO DỮ LIỆU] File đã được lưu an toàn tại {args.output_dir}")
    else:
        logger.warning("[CẢNH BÁO] Không thu thập được dữ liệu review nào.")

if __name__ == "__main__":
    main()
