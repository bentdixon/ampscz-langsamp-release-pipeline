from enum import Enum

# Match the language codes found in https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
# so that Stanza language identification is compatible
class Language(Enum):
    UNKNOWN = "UNKNOWN"
    zh = "Mandarin"
    es = "Spanish"
    en = "English"
    ko = "Korean"
    it = "Italian"
    ja = "Japanese"
    da = "Danish"
    de = "German"
    fr = "French"
    yue = "Cantonese"
    cn = "Chinese"  # NOT a valid Stanza language code - only a temporary fix for handling broken Chinese transcripts


SITE_CODE_TO_LANGUAGES: dict[str, tuple[Language, ...]] = {
    # Danish
    "CP": (Language.da,),
    # English
    "BI": (Language.en,),
    "CA": (Language.en,),
    "CM": (Language.en,),
    "GA": (Language.en,),
    "HA": (Language.en,),
    "NC": (Language.en,),
    "PA": (Language.en,),
    "SI": (Language.en,),
    "TE": (Language.en,),
    "WU": (Language.en,),
    "YA": (Language.en,),
    "IR": (Language.en,),
    "LA": (Language.en,),
    "KC": (Language.en,),
    "SF": (Language.en,),
    "SD": (Language.en,),
    "BM": (Language.en,),
    "SG": (Language.en,),
    "ME": (Language.en,),
    "CU": (Language.en,),
    "NL": (Language.en,),
    "NN": (Language.en,),
    "OH": (Language.en,),
    "OR": (Language.en,),
    "PI": (Language.en,),
    "UR": (Language.en,),
    # German
    "MU": (Language.de,),
    "JE": (Language.de,),
    # French
    "LS": (Language.fr,),
    # Italian
    "PV": (Language.it,),
    # Korean
    "GW": (Language.ko,),
    "SL": (Language.ko,),
    # Mandarin
    "SH": (Language.zh, Language.cn),
    # Spanish
    "MA": (Language.es,),
    "ST": (Language.es,),
    # Multilingual sites
    "MT": (Language.en, Language.fr),
    "HK": (Language.yue, Language.zh, Language.en, Language.cn),
    "CG": (Language.de, Language.en),
}


def get_site_languages(site_code: str) -> tuple[Language, ...]:
    """Return possible languages for a site code, or (UNKNOWN,) if not found."""
    return SITE_CODE_TO_LANGUAGES.get(site_code.upper(), (Language.UNKNOWN,))


def get_site_primary_language(site_code: str) -> Language:
    """Return the primary (first) language for a site code."""
    return get_site_languages(site_code)[0]