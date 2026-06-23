import re
import unicodedata

from ftfy import fix_text
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
from encoder import MODEL_NAME

LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
TEXT_COL = "text"
LABEL_COL = "label"
HR_LABEL_MERGE_MAP = {
    "DECOFEMPLYBEFORE": "DECOFEMPLY",
    "DECOFEMPLYAFTER": "DECOFEMPLY",
}


def load_hr_dataset(data_path: Path, min_tokens_limit: int = 15, min_class_samples: int = 50) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    hr_df = pd.read_csv(data_path)

    hr_df = hr_df[['content_en', 'label']].rename(columns={'content_en': TEXT_COL})

    hr_df[TEXT_COL] = hr_df[TEXT_COL].apply(_normalize_text)

    hr_df = hr_df.drop_duplicates(subset=[TEXT_COL])

    hr_df = hr_df[hr_df[TEXT_COL].apply(_is_english)]

    hr_df = hr_df[~hr_df[TEXT_COL].apply(_too_short(tokenizer, min_tokens_limit))]

    hr_df[LABEL_COL] = hr_df[LABEL_COL].replace(HR_LABEL_MERGE_MAP)

    hr_df = hr_df[~hr_df[LABEL_COL].isin(["PHOTO", "PASSPICTURE"])]

    hr_df = hr_df.groupby(LABEL_COL).filter(lambda group: len(group) >= min_class_samples)

    return hr_df


def load_wos_dataset(data_path: Path, min_class_samples: int = 50) -> pd.DataFrame:
    with open(data_path / "X.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines()]

    with open(data_path / "Y.txt", "r", encoding="utf-8") as f:
        label = [int(line.strip()) for line in f.readlines()]
        
    with open(data_path / "YL1.txt", "r", encoding="utf-8") as f:
        label_l1 = [int(line.strip()) for line in f.readlines()]
        
    with open(data_path / "YL2.txt", "r", encoding="utf-8") as f:
        label_l2 = [int(line.strip()) for line in f.readlines()]

    wos_df = pd.DataFrame(
        {
            TEXT_COL: docs,
            LABEL_COL: label,
            f"{LABEL_COL}_l1": label_l1,
            f"{LABEL_COL}_l2": label_l2,
        }
    )

    wos_df = wos_df.groupby(LABEL_COL).filter(lambda group: len(group) >= min_class_samples)

    return wos_df



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


def _too_short(text: str, tokenizer, limit: int) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True

    input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return len(input_ids) < limit