import os
import random
import json
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_dataset(num_records: int = 500, output_dir: str = "data/raw") -> str:
    """
    Sinh tập dữ liệu mẫu chuẩn thực tế cho 1 Seller (SmartLife Official) có 5 sản phẩm (BR04, BR05).
    Đảm bảo tính đa dạng văn phong (đa biến, không bị trùng lặp giả tạo), mô phỏng chân thực:
    - Review tích cực (khen chất lượng, giá, chăm sóc)
    - Review hỗn hợp / aspect trái chiều (khen máy, chê ship/đóng gói)
    - Review tiêu cực (lỗi kỹ thuật, vỡ móp, lừa đảo, shop vô trách nhiệm)
    - Review teencode, viết tắt, không dấu
    - Review spam nhận xu / câu mặc định sàn
    - Một tỷ lệ nhỏ review trùng lặp thực tế để kiểm thử BR12
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    shop_id = "shop_tech_888"
    products = [
        {"product_id": "SP01", "product_name": "Tai nghe Bluetooth True Wireless Pro Bass", "category": "Thiết bị âm thanh"},
        {"product_id": "SP02", "product_name": "Chuột Gaming Không Dây Silent Click RGB", "category": "Phụ kiện máy tính"},
        {"product_id": "SP03", "product_name": "Bàn Phím Cơ Không Dây 3 Chế Độ Kết Nối", "category": "Phụ kiện máy tính"},
        {"product_id": "SP04", "product_name": "Củ Sạc Nhanh GaN 65W 3 Cổng Tiện Lợi", "category": "Củ sạc & Cáp sạc"},
        {"product_id": "SP05", "product_name": "Cáp Sạc Nhanh Bọc Dù Chống Đứt 100W", "category": "Củ sạc & Cáp sạc"}
    ]

    sample_reviewers = [
        "Nguyễn Văn An", "Trần Thị Mai", "Lê Hoàng Long", "Phạm Đức Mạnh",
        "Hoàng Minh Tuấn", "Vũ Thùy Linh", "Đặng Quang Huy", "Bùi Thu Trang",
        "Đỗ Quốc Bảo", "Hồ Ngọc Khánh", "Ngô Thanh Hằng", "Dương Gia Bảo"
    ]

    # Các khối câu để ghép đa dạng
    pos_openers = [
        "Hàng nhận được rất ưng ý,", "Shop giao hàng siêu nhanh,", "Lần đầu mua của shop nhưng rất hài lòng,",
        "Sản phẩm chuẩn chính hãng,", "Quá tuyệt vời luôn shop ơi,", "Đóng gói cẩn thận 3 lớp chống sốc,",
        "Nhận hàng sau 2 ngày đặt,", "Chất lượng vượt xa mong đợi,"
    ]
    pos_bodies = [
        "dùng thử thấy pin rất trâu, kết nối mượt mà không bị delay",
        "chất liệu cao cấp cầm rất đầm tay, phím gõ êm ái",
        "sạc rất nhanh không hề bị nóng máy tí nào",
        "âm bass đập chắc nịch, mic đàm thoại to rõ ràng",
        "chuột bấm êm ru không phát ra tiếng click, dùng ban đêm cực thích",
        "thiết kế nhỏ gọn xinh xắn mang đi làm rất tiện lợi",
        "giá cả phải chăng mà chất lượng xịn xò 10 điểm"
    ]
    pos_closers = [
        "cho shop 5 sao nhé! 😍👍", "sẽ tiếp tục ủng hộ shop lần sau ❤️", "mọi người nên mua nha 💯",
        "rất đáng đồng tiền bát gạo ⭐⭐⭐⭐⭐", "chúc shop buôn may bán đắt 🎉"
    ]

    mixed_bodies = [
        "máy dùng rất ngon nhưng giao hàng hơi chậm hơn dự kiến 2 ngày 😐",
        "chất lượng sản phẩm okie nhưng hộp bên ngoài bị móp méo do vận chuyển",
        "tai nghe nghe nhạc hay nhưng gọi điện mic hơi nhỏ một chút",
        "chuột cầm vừa tay mà shop gửi nhầm màu trắng sang đen, cơ mà lười đổi",
        "củ sạc sạc nhanh tốt nhưng dùng lâu hơi ấm tay, giá này thì chấp nhận được",
        "bàn phím cơ gõ sướng tay, cơ mà shop rep tin nhắn tư vấn hơi chậm nha",
        "sản phẩm tốt trong tầm giá, nhưng bên vận chuyển quăng quật hộp hơi ghê"
    ]

    neg_openers = [
        "Quá thất vọng về cách làm việc,", "Hàng kém chất lượng nha mọi người,", "Dịch vụ sau bán hàng quá tệ,",
        "Mua về dùng được đúng 2 hôm thì hỏng,", "Lừa đảo, không giống như quảng cáo,",
        "Đóng gói quá ẩu cẩu thả,"
    ]
    neg_bodies = [
        "cắm sạc không lên nguồn, nhắn tin bảo hành thì shop im lặng không rep 😡",
        "tai nghe bên trái bị tịt hẳn không nghe thấy gì, yêu cầu shop hoàn tiền!",
        "chuột bị liệt nút cuộn, kết nối chập chờn liên tục rất ức chế",
        "bọc mỗi cái túi nilon làm bàn phím bị gãy mất một chân kê 😤",
        "giao hàng trễ hơn 10 ngày, hàng thì trầy xước như hàng cũ qua sử dụng",
        "dây cáp cắm vào chập chờn lúc nhận lúc không, phí cả tiền mua"
    ]
    neg_closers = [
        "1 sao không nói nhiều! 👎", "thất vọng toàn tập, ko bao giờ quay lại 😡",
        "mọi người né gấp shop này ra nhé 💔", "quá chán nản!"
    ]

    teencode_samples = [
        "sp rat tot nha sop, mik xai thay okela lam, ship nhanh vl luon",
        "k biet mn thay ntn chu mik thay on ap, gia re ma xai muot qua troi",
        "sp dep xiu, dung ok nhg ma ship hoi lau ti, shop nho ord dot sau nhanh hon nhe",
        "dg rat can than, kb xai ben k nhg trc mat thay rat ok nhe mn",
        "tot nha sop, se ung ho sop tiep vao lan sau nha hihi",
        "hang chuan auth do mn, mua dc gia sale re xiu, cam on shop nhieu nha",
        "dung tam on, nhung ma giao hang hoi cham ti teo, shipper thân thiện"
    ]

    spam_samples = [
        "hình ảnh và video mang tính chất nhận xu",
        "hình ảnh mang tính chất nhận xu ạ",
        "nhận xu là chính mua để lấy xu thôi",
        "chất lượng sản phẩm tuyệt vời",
        "đóng gói sản phẩm rất đẹp và chắc chắn",
        "thời gian giao hàng rất nhanh",
        "chưa dùng nên chưa biết khi nào dùng sẽ đánh giá sau",
        "chưa dùng thử nên chưa biết chất lượng thế nào",
        "aaaaabbbbbcccccdddd"
    ]

    records = []
    base_time = datetime(2026, 8, 1, 10, 0, 0)

    for i in range(1, num_records + 1):
        prod = random.choice(products)
        review_id = f"REV_{i:04d}"
        created_time = base_time + timedelta(days=random.randint(0, 70), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        created_at_str = created_time.strftime("%Y-%m-%d %H:%M:%S")

        # Có 6 bản ghi trùng lặp cố ý (BR12)
        if i in [25, 26]:
            text = "Tai nghe xịn xò lắm nha mn, bass đập căng đét, đóng gói rất cẩn thận, cho shop 5 sao 😍👍"
            rating = 5
        elif i in [80, 81]:
            text = "Máy dùng rất tốt nhưng giao hàng hơi lâu, đóng gói hơi móp hộp xíu 😐"
            rating = 4
        elif i in [150, 151]:
            text = "Hàng kém chất lượng, cắm sạc không lên nguồn, gọi shop không nghe máy 😡👎"
            rating = 1
        else:
            dice = random.random()
            if dice < 0.50:
                # 50% Tích cực
                text = f"{random.choice(pos_openers)} {random.choice(pos_bodies)}, {random.choice(pos_closers)}"
                rating = 5 if random.random() < 0.85 else 4
            elif dice < 0.70:
                # 20% Hỗn hợp / Aspect trái chiều
                text = f"{random.choice(mixed_bodies)}"
                rating = random.choice([3, 4, 4, 5])
            elif dice < 0.85:
                # 15% Tiêu cực
                text = f"{random.choice(neg_openers)} {random.choice(neg_bodies)} {random.choice(neg_closers)}"
                rating = random.choice([1, 1, 2, 3])
            elif dice < 0.92:
                # 7% Teencode
                text = random.choice(teencode_samples)
                rating = random.choice([4, 5])
            else:
                # 8% Spam
                text = random.choice(spam_samples)
                rating = 5

        # Phân bố lượt like: 70% 0 like, 20% 1-5 like, 10% 6-25 like
        like_dice = random.random()
        if like_dice < 0.70:
            likes = 0
        elif like_dice < 0.90:
            likes = random.randint(1, 5)
        else:
            likes = random.randint(6, 25)

        reviewer = random.choice(sample_reviewers)

        records.append({
            "review_id": review_id,
            "shop_id": shop_id,
            "product_id": prod["product_id"],
            "product_name": prod["product_name"],
            "category": prod["category"],
            "rating": rating,
            "comment_text": text,
            "created_at": created_at_str,
            "reviewer_name": reviewer,
            "like_count": likes,
            "platform": "mock_shopee_crawler"
        })

    csv_path = os.path.join(output_dir, "sample_raw.csv")
    json_path = os.path.join(output_dir, "sample_raw.json")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[THÀNH CÔNG] Đã sinh {num_records} review mẫu phong phú:")
    print(f" -> CSV:  {csv_path}")
    print(f" -> JSON: {json_path}")
    return csv_path

if __name__ == "__main__":
    generate_sample_dataset()
