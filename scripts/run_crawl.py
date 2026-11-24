import argparse
import sys
import os

# Thêm đường dẫn gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawler.tiki_crawler import TikiCrawler
from src.crawler.shopee_crawler import ShopeeCrawler
from src.storage.sql_server_store import SQLServerStore, sqlserver_is_configured
from src.utils.logger import get_logger

logger = get_logger("run_crawl")


def main():
    parser = argparse.ArgumentParser(description="Công cụ cào dữ liệu đánh giá sàn TMĐT (Shopee / Tiki)")
    parser.add_argument("--platform", type=str, choices=["tiki", "shopee"], default="tiki", help="Nền tảng TMĐT cần cào")
    parser.add_argument("--product_id", type=str, required=True, help="Mã sản phẩm (product_id hoặc item_id)")
    parser.add_argument("--shop_id", type=str, default="auto", help="Mã cửa hàng (shop_id), mặc định auto")
    parser.add_argument("--product_name", type=str, default=None, help="Tên sản phẩm (tự động lấy nếu bỏ trống)")
    parser.add_argument("--category", type=str, default=None, help="Danh mục sản phẩm (tự động phát hiện nếu bỏ trống)")
    parser.add_argument("--max_reviews", type=int, default=100, help="Số lượng review tối đa cần cào (nhập 0 hoặc dùng --all để cào hết)")
    parser.add_argument("--all", action="store_true", help="Cào TOÀN BỘ review của sản phẩm không giới hạn số lượng")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Thư mục lưu trữ raw data")
    parser.add_argument("--no_preprocess", action="store_true", help="Không tự động chạy tiền xử lý sau khi cào")
    parser.add_argument("--csv_only", action="store_true", help="Chỉ lưu CSV/JSON, không ghi vào SQL Server")

    args = parser.parse_args()

    max_reviews = 0 if args.all else args.max_reviews

    logger.info(f"Khởi động Crawler cho sàn: {args.platform.upper()} (Chế độ: {'Cào toàn bộ' if max_reviews == 0 else f'Tối đa {max_reviews}'})")

    if args.platform == "shopee" and args.shop_id == "auto":
        logger.error("Shopee API yêu cầu phải cung cấp --shop_id cụ thể của cửa hàng!")
        sys.exit(1)

    sql_store = None
    crawl_run_id = None
    reviews_upserted = 0
    crawler = None
    try:
        if not args.csv_only and sqlserver_is_configured():
            sql_store = SQLServerStore()
            sql_store.ping()
            crawl_run_id = sql_store.start_crawl_run(
                platform=args.platform,
                product_id=args.product_id,
                shop_id=args.shop_id,
                product_name=args.product_name or "",
                category=args.category or "",
                requested_reviews=max_reviews,
            )
            logger.info(f"Đã tạo crawl run SQL Server: {crawl_run_id}")
        elif not args.csv_only:
            logger.info("SQLSERVER_SERVER chưa được cấu hình; dùng chế độ CSV/JSON hiện tại.")

        if args.platform == "tiki":
            crawler = TikiCrawler()
            crawler.fetch_product_reviews(
                shop_id=args.shop_id,
                product_id=args.product_id,
                product_name=args.product_name,
                max_reviews=max_reviews,
                category=args.category,
            )
        else:
            crawler = ShopeeCrawler()
            crawler.fetch_product_reviews(
                shop_id=args.shop_id,
                product_id=args.product_id,
                product_name=args.product_name or "Sản phẩm Shopee",
                max_reviews=max_reviews,
                category=args.category or "Thiết bị điện tử",
            )

        # Keep CSV/JSON output for offline processing and compatibility.
        saved_files = crawler.save_raw_data(output_dir=args.output_dir)
        if sql_store:
            product = crawler.product_info or {
                "platform": args.platform,
                "product_id": args.product_id,
                "shop_id": args.shop_id,
                "product_name": args.product_name or f"Sản phẩm {args.product_id}",
                "category": args.category or "Chưa phân loại",
            }
            sql_store.upsert_product(product)
            reviews_upserted = sql_store.upsert_reviews(
                crawler.reviews,
                crawl_run_id=crawl_run_id,
            )
            sql_store.finish_crawl_run(
                run_id=crawl_run_id,
                status="completed" if crawler.reviews else "no_reviews",
                reviews_collected=len(crawler.reviews),
                reviews_upserted=reviews_upserted,
                product=product,
            )

        if saved_files:
            logger.info(f"[HOÀN TẤT CÀO DỮ LIỆU] Dữ liệu đã đồng bộ vào bộ '{args.platform}': {saved_files['csv']}")

            if not args.no_preprocess:
                logger.info("Tự động kích hoạt tiền xử lý dữ liệu để cập nhật Dashboard...")
                try:
                    from scripts.run_preprocess import run_full_preprocessing
                    run_full_preprocessing(
                        raw_dir=args.output_dir,
                        sync_sql_server=not args.csv_only,
                    )
                except Exception as error:
                    logger.warning(f"Lỗi khi tự động chạy tiền xử lý: {error}. Bạn có thể chạy thủ công: python scripts/run_preprocess.py")
        else:
            logger.warning("[CẢNH BÁO] Không thu thập được dữ liệu review nào.")
    except Exception as error:
        if sql_store and crawl_run_id:
            try:
                sql_store.finish_crawl_run(
                    run_id=crawl_run_id,
                    status="failed",
                    reviews_collected=len(crawler.reviews) if crawler else 0,
                    reviews_upserted=reviews_upserted,
                    error=str(error),
                )
            except Exception as status_error:
                logger.error(f"Không thể cập nhật trạng thái crawl run {crawl_run_id}: {status_error}")
        raise
    finally:
        if sql_store:
            sql_store.close()

if __name__ == "__main__":
    main()
