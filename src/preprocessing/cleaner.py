import re
import unicodedata

class TextCleaner:
    """
    Module làm sạch văn bản đánh giá tiếng Việt.
    Nhiệm vụ:
    - Loại bỏ thẻ HTML, URL, email
    - Chuẩn hóa Unicode dựng sẵn (NFC)
    - Rút gọn ký tự lặp quá mức (kéo dài cảm xúc: "đẹpppp", "ngonnnnn")
    - Chuẩn hóa khoảng trắng
    """

    HTML_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    # Ký tự lặp từ 3 lần trở lên (cả tiếng Anh và tiếng Việt có dấu)
    REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ])\1{2,}", re.IGNORECASE)

    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """
        Chuẩn hóa Unicode dạng dựng sẵn (NFC).
        Giải quyết triệt để lỗi gõ Telex/VNI bị lệch bảng mã tổ hợp tiếng Việt.
        """
        if not text:
            return ""
        return unicodedata.normalize("NFC", str(text))

    @classmethod
    def remove_html_and_urls(cls, text: str) -> str:
        """Xóa thẻ HTML và các đường link URL trong review."""
        if not text:
            return ""
        text = cls.HTML_PATTERN.sub(" ", text)
        text = cls.URL_PATTERN.sub(" ", text)
        text = cls.EMAIL_PATTERN.sub(" ", text)
        return text

    @classmethod
    def reduce_repeated_characters(cls, text: str) -> str:
        """
        Rút gọn các ký tự bị kéo dài cảm xúc:
        'đẹpppp' -> 'đẹp', 'ngonnnnn' -> 'ngon', 'quáaaaa' -> 'quá'
        """
        if not text:
            return ""
        return cls.REPEATED_CHAR_PATTERN.sub(r"\1", text)

    @classmethod
    def clean_whitespace(cls, text: str) -> str:
        """Xóa khoảng trắng thừa và dòng trống."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def clean(cls, text: str) -> str:
        """Thực thi chuỗi thao tác làm sạch căn bản."""
        if not text:
            return ""
        text = cls.normalize_unicode(text)
        text = cls.remove_html_and_urls(text)
        text = cls.reduce_repeated_characters(text)
        text = cls.clean_whitespace(text)
        return text
