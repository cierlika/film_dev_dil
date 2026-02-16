"""
film_metadata.py
----------------
Static lookup tables and feature builder for film metadata:
  - film_manufacturer  (numeric-encoded)
  - film_family        (numeric-encoded)
  - box_iso_category   (ordinal integer)

No target encoding is used – all mappings are deterministic reference data,
so there is zero risk of data leakage and no fit() step is required.
"""

import re
import numpy as np
import pandas as pd

# ── Film family rules ─────────────────────────────────────────────────────────
# Ordered list of (substring_lower, family_label) pairs.
# First match wins.  Films that match nothing → "traditional"
#
# Families:
#   t_grain      – tabular-grain emulsions (T-Max, Delta, Acros)
#   chromogenic  – C-41 B&W process (XP2, 400CN)
#   infrared     – IR-sensitive films
#   traditional  – standard cubic-grain (HP5, Tri-X, Fomapan, …)

FILM_FAMILY_RULES: list[tuple[str, str]] = [
    # ── T-grain / tabular-grain ──────────────────────────────────────────────
    ("t-max",           "t_grain"),
    ("tmax",            "t_grain"),
    ("t max",           "t_grain"),
    (" tmx",            "t_grain"),
    (" tmy",            "t_grain"),
    ("delta 1",         "t_grain"),   # Ilford Delta 100
    ("delta 4",         "t_grain"),   # Ilford Delta 400
    ("delta 32",        "t_grain"),   # Ilford Delta 3200
    ("ilford delta",    "t_grain"),
    ("acros",           "t_grain"),   # Fuji Acros / Acros II
    ("neopan acros",    "t_grain"),
    # ── Chromogenic (C-41) ───────────────────────────────────────────────────
    ("xp2",             "chromogenic"),
    ("400cn",           "chromogenic"),
    ("bw400cn",         "chromogenic"),
    ("bwc",             "chromogenic"),
    # ── Infrared ─────────────────────────────────────────────────────────────
    ("infrared",        "infrared"),
    (" ir ",            "infrared"),
    ("-ir",             "infrared"),
    (" ir$",            "infrared"),  # trailing " ir"
    ("rollei ir",       "infrared"),
    ("efke ir",         "infrared"),
    ("kodak hie",       "infrared"),
    ("hie",             "infrared"),
    ("sfx",             "infrared"),
    ("maco ir",         "infrared"),
    ("ilford sfx",      "infrared"),
    # ── Ortho / document ────────────────────────────────────────────────────
    ("copex",           "ortho"),
    ("ortho plus",      "ortho"),
    ("ortho",           "ortho"),
    # ── Everything else is traditional cubic-grain ───────────────────────────
]

FILM_FAMILY_CODES: dict[str, int] = {
    "t_grain":     0,
    "chromogenic": 1,
    "infrared":    2,
    "ortho":       3,
    "traditional": 4,   # fallback / default
}

# ── Manufacturer rules ────────────────────────────────────────────────────────
# Ordered list of (substring_lower, manufacturer_label) pairs.
# First match wins.  Unknown films → "Unknown"

MANUFACTURER_RULES: list[tuple[str, str]] = [
    ("kodak",           "Kodak"),
    ("tri-x",           "Kodak"),      # Tri-X always Kodak
    ("plus-x",          "Kodak"),
    ("double-x",        "Kodak"),
    ("verichrome",      "Kodak"),
    ("panatomic",       "Kodak"),
    ("ilford",          "Ilford"),
    ("hp5",             "Ilford"),
    ("fp4",             "Ilford"),
    ("pan f",           "Ilford"),
    ("panf",            "Ilford"),
    ("delta",           "Ilford"),
    ("xp2",             "Ilford"),
    ("kentmere",        "Ilford"),     # Ilford-owned brand
    ("fujifilm",        "Fuji"),
    ("fuji",            "Fuji"),
    ("neopan",          "Fuji"),
    ("acros",           "Fuji"),
    ("fomapan",         "Foma"),
    ("foma",            "Foma"),
    ("retro 80s",       "Rollei"),
    ("retro 400s",      "Rollei"),
    ("rollei",          "Rollei"),
    ("agfaphoto",       "Agfa"),
    ("agfa",            "Agfa"),
    ("apx",             "Agfa"),
    ("adox",            "Adox"),
    ("efke",            "Adox"),       # Adox/Efke overlap
    ("chs",             "Adox"),       # Adox CHS
    ("bergger",         "Bergger"),
    ("pancro",          "Bergger"),
    ("orwo",            "ORWO"),
    ("svema",           "Svema"),
    ("shanghai",        "Shanghai"),
    ("lucky",           "Lucky"),
    ("ferrania",        "Ferrania"),
    ("p30",             "Ferrania"),
    ("cinestill",       "CineStill"),
    ("lomography",      "Lomography"),
    ("lomo",            "Lomography"),
    ("earl grey",       "Lomography"),
    ("lady grey",       "Lomography"),
    ("arista",          "Arista"),
    ("freestyle",       "Freestyle"),
    ("forte",           "Forte"),
    ("maco",            "Maco"),
    ("washi",           "FilmWashi"),
    ("ultrafine",       "Ultrafine"),
    ("polypan",         "Polypan"),
    ("kosmo",           "KosmoFoto"),
    ("photo memo",      "Silberra"),
    ("silberra",        "Silberra"),
    ("flic film",       "FlicFilm"),
    ("film washi",      "FilmWashi"),
]

MANUFACTURER_CODES: dict[str, int] = {}  # built dynamically in FilmMetadataBuilder

# ── ISO category bins ─────────────────────────────────────────────────────────
# Using box_iso (already present from FilmSlopeEstimator).
# Bins match the summary in the improvement plan.

ISO_BINS:   list[float] = [0,   25,  64,  160,  320,  640,  1280, 3200, float("inf")]
ISO_LABELS: list[str]   = [
    "iso_sub25",      # ≤25   (Pan F, Acros 100 at box, very slow stocks)
    "iso_25_64",      # 26-64
    "iso_64_160",     # 65-160  (125, 100)
    "iso_160_320",    # 161-320 (200)
    "iso_320_640",    # 321-640 (400)
    "iso_640_1280",   # 641-1280 (800)
    "iso_1280_3200",  # 1281-3200
    "iso_3200plus",   # >3200
]
ISO_LABEL_CODES: dict[str, int] = {lbl: i for i, lbl in enumerate(ISO_LABELS)}


def _match_rules(value: str, rules: list[tuple[str, str]], default: str) -> str:
    """Return the label of the first matching rule, or *default*."""
    if not isinstance(value, str):
        return default
    lower = value.lower()
    for pattern, label in rules:
        # Use regex for patterns ending with $ to support end-of-string anchors
        if "$" in pattern:
            if re.search(pattern, lower):
                return label
        elif pattern in lower:
            return label
    return default


# ── Builder class ─────────────────────────────────────────────────────────────

class FilmMetadataBuilder:
    """
    Adds three static metadata columns to a DataFrame that has a 'Film' column
    and a 'box_iso' column (from FilmSlopeEstimator):

        film_manufacturer_code  – integer-encoded manufacturer
        film_family_code        – integer-encoded emulsion family
        box_iso_category_code   – ordinal ISO speed bin (0 = slowest)

    There is no fit() step because all mappings are deterministic reference data.
    Call transform() on both train and test without fitting.
    """

    # Collect all unique manufacturer labels and assign stable codes
    _MANUFACTURER_LABELS: list[str] = sorted(
        {label for _, label in MANUFACTURER_RULES}
    ) + ["Unknown"]
    _MFR_CODE: dict[str, int] = {
        lbl: i for i, lbl in enumerate(_MANUFACTURER_LABELS)
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a NEW DataFrame (same index) with the three metadata columns.
        Does not modify the input.
        """
        out = pd.DataFrame(index=df.index)

        film_lower = df["Film"].fillna("").str.lower()

        # ── manufacturer ──────────────────────────────────────────────────
        manufacturers = film_lower.map(
            lambda s: _match_rules(s, MANUFACTURER_RULES, "Unknown")
        )
        out["film_manufacturer_code"] = manufacturers.map(self._MFR_CODE).fillna(
            self._MFR_CODE["Unknown"]
        ).astype(int)

        # ── film family ───────────────────────────────────────────────────
        families = film_lower.map(
            lambda s: _match_rules(s, FILM_FAMILY_RULES, "traditional")
        )
        out["film_family_code"] = families.map(FILM_FAMILY_CODES).fillna(
            FILM_FAMILY_CODES["traditional"]
        ).astype(int)

        # ── ISO category ──────────────────────────────────────────────────
        if "box_iso" in df.columns:
            cats = pd.cut(
                df["box_iso"],
                bins=ISO_BINS,
                labels=ISO_LABELS,
                right=True,
            )
            out["box_iso_category_code"] = cats.map(ISO_LABEL_CODES).fillna(-1).astype(int)
        else:
            out["box_iso_category_code"] = -1

        return out

    # ── Convenience: decode codes back to labels (useful for debugging) ──────
    @staticmethod
    def decode_manufacturer(code: int) -> str:
        labels = FilmMetadataBuilder._MANUFACTURER_LABELS
        if 0 <= code < len(labels):
            return labels[code]
        return "Unknown"

    @staticmethod
    def decode_family(code: int) -> str:
        rev = {v: k for k, v in FILM_FAMILY_CODES.items()}
        return rev.get(code, "unknown")

    @staticmethod
    def decode_iso_category(code: int) -> str:
        if 0 <= code < len(ISO_LABELS):
            return ISO_LABELS[code]
        return "unknown"
