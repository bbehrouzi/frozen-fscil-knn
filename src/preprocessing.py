import pandas as pd
import re
import unicodedata
from pathlib import Path

from ftfy import fix_text
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
WOS_DATA_PATH = ROOT_DIR / "data" / "WebOfScience" / "WOS11967"
HR_DATA_PATH = ROOT_DIR / "data" / "Ankaadia" / "data_2025_10_26.csv"
LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MIN_TOKENS_LIMIT = 15
MIN_CLASS_SAMPLES = 50
HR_LABEL_MERGE_MAP = {
    "DECOFEMPLYBEFORE": "DECOFEMPLY",
    "DECOFEMPLYAFTER": "DECOFEMPLY",
}


def _normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = fix_text(str(text))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_english(
    text: str,
    english_threshold: float = 0.80,
    non_english_threshold: float = 0.90,
    min_chars: int = 30,
) -> bool:
    if len(text) < min_chars:
        return True

    try:
        confidence_values = LANG_DETECTOR.compute_language_confidence_values(text[:1000])
    except Exception:
        return True

    if not confidence_values:
        return True

    top = confidence_values[0]
    en_score = 0.0
    for value in confidence_values:
        if value.language == Language.ENGLISH:
            en_score = value.value
            break

    if en_score >= english_threshold:
        return True

    if top.language != Language.ENGLISH and top.value >= non_english_threshold:
        return False

    return True


def _too_short(text: str, limit: int) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True

    input_ids = TOKENIZER.encode(text, add_special_tokens=False, truncation=False)
    return len(input_ids) < limit


def get_hr_dataframe() -> pd.DataFrame:
    hr_df = pd.read_csv(HR_DATA_PATH)

    hr_df = hr_df[['content_en', 'label']].rename(columns={'content_en': 'docs'})
    hr_df["docs"] = hr_df["docs"].apply(_normalize_text)
    hr_df = hr_df.drop_duplicates(subset=["docs"])
    print(len(hr_df))

    len_before = len(hr_df)
    hr_df = hr_df[hr_df["docs"].apply(_is_english)]
    print(f"Number of non-english docs removed: {len_before - len(hr_df)}")

    len_before = len(hr_df)
    hr_df = hr_df[~hr_df["docs"].apply(_too_short, limit=MIN_TOKENS_LIMIT)]
    print(f"Number of short docs removed: {len_before - len(hr_df)}")

    hr_df["label"] = hr_df["label"].replace(HR_LABEL_MERGE_MAP)
    hr_df = hr_df[~hr_df["label"].isin(["PHOTO", "PASSPICTURE"])]

    hr_df = hr_df.groupby("label").filter(lambda group: len(group) >= MIN_CLASS_SAMPLES)

    return hr_df


def get_wos_dataframe() -> pd.Dataframe:
    with open(WOS_DATA_PATH / "X.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines()]

    with open(WOS_DATA_PATH / "Y.txt", "r", encoding="utf-8") as f:
        label = [int(line.strip()) for line in f.readlines()]
        
    with open(WOS_DATA_PATH / "YL1.txt", "r", encoding="utf-8") as f:
        label_l1 = [int(line.strip()) for line in f.readlines()]
        
    with open(WOS_DATA_PATH / "YL2.txt", "r", encoding="utf-8") as f:
        label_l2 = [int(line.strip()) for line in f.readlines()]

    wos_df = pd.DataFrame(
        {
            "docs": docs,
            "label": label,
            "label_l1": label_l1,
            "label_l2": label_l2,
        }
    )

    wos_df = wos_df.groupby("label").filter(lambda group: len(group) >= MIN_CLASS_SAMPLES)

    return wos_df
