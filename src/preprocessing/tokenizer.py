from src.utils.logger import get_logger

logger = get_logger("tokenizer")

# Ưu tiên underthesea (ổn định, không kén C++ DLL trên Windows), fallback pyvi
USE_UNDERTHESEA = False
USE_PYVI = False

try:
    from underthesea import word_tokenize
    USE_UNDERTHESEA = True
    logger.info("Sử dụng engine tách từ tiếng Việt: underthesea")
except ImportError:
    try:
        from pyvi import ViTokenizer
        USE_PYVI = True
        logger.info("Sử dụng engine tách từ tiếng Việt: pyvi")
    except Exception:
        logger.warning("Không tìm thấy underthesea hoặc pyvi hợp lệ. Sẽ dùng bộ tách từ mặc định.")

class VietnameseTokenizer:
    """
    Module phân đoạn từ tiếng Việt (Word Segmentation).
    Ví dụ: 'giao hàng rất nhanh' -> 'giao_hàng rất nhanh'
    Giúp các mô hình như TF-IDF, PhoBERT, SVM nhận biết các từ ghép ngữ nghĩa.
    """

    @classmethod
    def tokenize(cls, text: str) -> str:
        if not text:
            return ""

        if USE_UNDERTHESEA:
            try:
                # format="text" nối các từ ghép bằng dấu gạch dưới "_"
                tokens = word_tokenize(text, format="text")
                return tokens
            except Exception as e:
                logger.debug(f"Lỗi khi tách từ bằng underthesea: {e}")
                return text

        if USE_PYVI:
            try:
                return ViTokenizer.tokenize(text)
            except Exception as e:
                logger.debug(f"Lỗi khi tách từ bằng pyvi: {e}")
                return text

        return text
