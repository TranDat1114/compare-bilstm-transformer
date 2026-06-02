"""Tiền xử lý văn bản tiếng Việt cho bài toán phát hiện ngôn từ thù ghét (ViHSD).

Dữ liệu là bình luận mạng xã hội -> nhiều nhiễu đặc thù tiếng Việt:
  - Teencode / viết tắt:  ko, k, hok -> không ; dc, đc -> được ; vs -> với ...
  - Emoji & emoticon:  ":))", "=))", "❤️", "🤣" ... mang sắc thái (mỉa mai, cười cợt).
  - Kéo dài ký tự:  "quááá", "hayyyy", "=))))" (elongation) -> cùng một ý.
  - URL, @mention, hashtag, số điện thoại.
  - Unicode tổ hợp (NFC vs NFD) cho dấu tiếng Việt.

Mỗi bước đều có LÝ DO và được dùng cho phần "ablation" trong README:
nếu BỎ bước này thì OOV tăng / tín hiệu cảm xúc mất / vốn từ phân mảnh.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------
# 1) Từ điển teencode -> dạng chuẩn.  (Rút gọn nhưng phủ các trường hợp phổ biến
#    nhất trong tập train, ví dụ "ko" xuất hiện 1363 lần, "dc"/"đc" rất nhiều.)
# ----------------------------------------------------------------------------
TEENCODE = {
    "ko": "không", "k": "không", "kg": "không", "khong": "không", "hok": "không",
    "hong": "không", "hông": "không", "kh": "không", "khg": "không", "ks": "không",
    "dc": "được", "đc": "được", "duoc": "được", "dk": "được",
    "vs": "với", "vss": "với", "voi": "với",
    "ng": "người", "ngta": "người ta", "nguoi": "người",
    "j": "gì", "z": "vậy", "zậy": "vậy", "v": "vậy", "vay": "vậy",
    "bt": "bình thường", "bth": "bình thường",
    "mn": "mọi người", "mng": "mọi người",
    "ae": "anh em", "ace": "anh chị em",
    "ck": "chồng", "vk": "vợ",
    "trc": "trước", "sau": "sau", "s": "sao", "sao": "sao",
    "r": "rồi", "rui": "rồi", "roi": "rồi",
    "h": "giờ", "h.": "giờ",
    "t": "tao", "m": "mày", "tui": "tôi", "tao": "tao", "may": "mày",
    "đt": "điện thoại", "dt": "điện thoại",
    "ny": "người yêu", "fb": "facebook", "face": "facebook",
    "vcl": "vãi", "vl": "vãi", "vkl": "vãi", "vc": "vãi",
    "đm": "đm", "dm": "đm", "đmm": "đm", "vch": "vãi",
    "nt": "nhắn tin", "ib": "nhắn tin", "inbox": "nhắn tin",
    "ô": "ông", "ô.": "ông", "b": "bạn", "bn": "bạn", "bn.": "bạn",
    "thик": "thích", "ny.": "người yêu", "wa": "quá", "wá": "quá", "qá": "quá",
    "trg": "trong", "tjnh": "tình", "cx": "cũng", "cũg": "cũng",
    "ny": "người yêu", "lm": "làm", "lèm": "làm", "ms": "mới", "đg": "đang",
    "nta": "người ta", "mik": "mình", "mìk": "mình", "mh": "mình",
    "thui": "thôi", "gud": "tốt", "good": "tốt", "ok": "ổn", "oke": "ổn", "okê": "ổn",
}

# ----------------------------------------------------------------------------
# 2) Emoticon ASCII -> token cảm xúc.  Giữ lại tín hiệu (rất quan trọng với mỉa mai).
# ----------------------------------------------------------------------------
EMOTICON = [
    (r"=\)+|:\)+|:-\)|\^_\^|\^\^|:d|:\>|=d|c:", " <pos_emo> "),
    (r":\(+|=\(+|:-\(|t_t|:'\(|:<|;\(", " <neg_emo> "),
    (r":v|:3|:p|:b|=p|xd|=]+|:]+", " <troll_emo> "),
    (r"<3|❤|♥", " <heart_emo> "),
]

# Dải Unicode chứa emoji (mặt cười, biểu tượng, bàn tay, ...).
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "]",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PHONE_RE = re.compile(r"\b0\d{8,10}\b")
# Lặp ký tự >= 3 lần -> rút về 1. Loại trừ dấu câu (!?.,) để bước MULTI_PUNCT
# xử lý chúng riêng, tránh elongation "ăn" mất chuỗi dấu câu.
ELONG_RE = re.compile(r"([^!?.,])\1{2,}")
# Chuẩn hoá chuỗi dấu câu lặp ("!!!", "???") thành MỘT token dấu câu -> giảm
# độ thưa của vốn từ mà vẫn giữ tín hiệu "có dấu cảm/hỏi".
MULTI_PUNCT_RE = re.compile(r"([!?.,])\1+")
SPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Chuẩn hoá NFC: gộp ký tự + dấu thành 1 mã điểm (ổn định cho tiếng Việt)."""
    return unicodedata.normalize("NFC", text)


def replace_emoji(text: str) -> str:
    return EMOJI_RE.sub(" <emoji> ", text)


def replace_emoticon(text: str) -> str:
    for pat, tok in EMOTICON:
        text = re.sub(pat, tok, text, flags=re.IGNORECASE)
    return text


def reduce_elongation(text: str) -> str:
    """'quááá' -> 'quá', 'hayyy' -> 'hay'."""
    return ELONG_RE.sub(r"\1", text)


def normalize_teencode(tokens):
    return [TEENCODE.get(t, t) for t in tokens]


def basic_segment(text: str):
    """Tách token theo khoảng trắng, tách dấu câu ra khỏi từ."""
    text = re.sub(r"([!?.,;:()\"])", r" \1 ", text)
    return text.split()


def clean_text(text: str, lowercase: bool = True, do_teencode: bool = True) -> str:
    """Pipeline đầy đủ. Trả về chuỗi đã làm sạch (token cách nhau bởi 1 space)."""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    if lowercase:
        text = text.lower()
    text = URL_RE.sub(" <url> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = replace_emoticon(text)          # trước khi xoá ký tự đặc biệt
    text = replace_emoji(text)
    text = reduce_elongation(text)
    text = MULTI_PUNCT_RE.sub(r" \1 ", text)
    # bỏ ký tự không phải chữ/số tiếng Việt/token đặc biệt, giữ <...>
    text = re.sub(r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ<>_!?.,\s]", " ", text)
    tokens = basic_segment(text)
    if do_teencode:
        tokens = normalize_teencode(tokens)
    text = " ".join(tokens)
    return SPACE_RE.sub(" ", text).strip()


def simple_tokenize(text: str, lowercase: bool = True) -> str:
    """Đường cơ sở KHÔNG làm sạch (dùng cho ablation 'no-clean'):
    chỉ chuẩn hoá unicode + lowercase + tách khoảng trắng."""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    if lowercase:
        text = text.lower()
    return SPACE_RE.sub(" ", text).strip()


# Tách từ tiếng Việt (pyvi) — tuỳ chọn cho ablation 'word_segment'.
_VI_TOKENIZER = None


def word_segment(text: str) -> str:
    global _VI_TOKENIZER
    if _VI_TOKENIZER is None:
        from pyvi import ViTokenizer
        _VI_TOKENIZER = ViTokenizer
    # pyvi nối các âm tiết cùng 1 từ bằng '_': "học sinh" -> "học_sinh"
    return _VI_TOKENIZER.tokenize(text)


def build_preprocessor(clean: bool = True, lowercase: bool = True,
                       word_seg: bool = False):
    """Trả về hàm str->str theo cấu hình (dùng thống nhất cho train/dev/test)."""
    def _fn(text: str) -> str:
        if clean:
            text = clean_text(text, lowercase=lowercase)
        else:
            text = simple_tokenize(text, lowercase=lowercase)
        if word_seg:
            text = word_segment(text)
        return text
    return _fn


if __name__ == "__main__":
    samples = [
        "Em được làm fan cứng luôn rồi nè ❤️ reaction quá hay quá cute coi mấy giờ này =]]]",
        "ko hiểu sao bọn nó cứ chửi nhau =))))) vcl",
        "Đm thằng này ngu vl :)) đọc link https://abc.xyz đi @user",
    ]
    for s in samples:
        print("RAW :", s)
        print("CLEAN:", clean_text(s))
        print()
