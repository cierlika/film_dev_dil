"""
developer_metadata.py
---------------------
Static lookup table and feature builder for developer type classification.

Developer types:
  general_purpose  – D-76, ID-11, HC-110, Ilfosol, XTOL, …
  fine_grain       – Perceptol, Microdol-X, …
  speed_enhancing  – Microphen, DD-X, Diafine, …
  compensating     – Rodinal, R09, Adonal, …
  tabular_grain    – T-Max Developer, T-Max RS, …
  staining         – Pyro developers (PMK, Pyrocat-HD, …)
  alternative      – Caffenol, ascorbic-acid based, …

No target encoding → no fit() step → zero leakage risk.
"""

import pandas as pd

# ── Developer type rules ──────────────────────────────────────────────────────
# Ordered list of (lowercase_substring, type_label).
# IMPORTANT: more specific entries must come before shorter substrings of the
# same developer name (e.g. "pyrocat-hd" before "pyrocat" before "pyro").
# First match wins.  Unknown developers → "general_purpose".

DEVELOPER_TYPE_RULES: list[tuple[str, str]] = [

    # ── Tabular-grain optimised ──────────────────────────────────────────────
    ("t-max rs",            "tabular_grain"),
    ("tmax rs",             "tabular_grain"),
    ("t max rs",            "tabular_grain"),
    ("t-max dev",           "tabular_grain"),
    ("tmax dev",            "tabular_grain"),
    ("t max dev",           "tabular_grain"),
    ("tmax developer",      "tabular_grain"),
    ("t-max developer",     "tabular_grain"),
    ("kodak xtol",          "tabular_grain"),
    ("xtol",                "tabular_grain"),

    # ── Staining / pyro ──────────────────────────────────────────────────────
    ("pyrocat-hd",          "staining"),
    ("pyrocat hd",          "staining"),
    ("pyrocat-mc",          "staining"),
    ("pyrocat mc",          "staining"),
    ("pyrocat",             "staining"),
    ("pmk pyro",            "staining"),
    ("pmk",                 "staining"),
    ("510-pyro",            "staining"),
    ("510 pyro",            "staining"),
    ("abc pyro",            "staining"),
    ("harvey's 777",        "staining"),
    ("formulary wo-2",      "staining"),
    ("wd2d",                "staining"),
    ("wb2d",                "staining"),
    ("pyro",                "staining"),   # catches "pyro" as standalone word too

    # ── Alternative / DIY ────────────────────────────────────────────────────
    ("caffenol-c-h",        "alternative"),
    ("caffenol-c-l",        "alternative"),
    ("caffenol-c-m",        "alternative"),
    ("caffenol-c",          "alternative"),
    ("caffenol",            "alternative"),
    ("vitamin c",           "alternative"),
    ("ascorbic",            "alternative"),
    ("parodinal",           "alternative"),

    # ── Fine grain ───────────────────────────────────────────────────────────
    ("perceptol",           "fine_grain"),
    ("microdol-x",          "fine_grain"),
    ("microdol x",          "fine_grain"),
    ("microdol",            "fine_grain"),
    ("spur acurol-n",       "fine_grain"),
    ("spur acurol",         "fine_grain"),
    ("spur df-96",          "fine_grain"),
    ("tetenal ultrafin special", "fine_grain"),
    ("promicrol",           "fine_grain"),
    ("edwal fg-7",          "fine_grain"),
    ("edwal fg7",           "fine_grain"),
    ("spur hrd",            "fine_grain"),

    # ── Speed enhancing ──────────────────────────────────────────────────────
    ("microphen",           "speed_enhancing"),
    ("dd-x",                "speed_enhancing"),
    ("ddx",                 "speed_enhancing"),
    ("diafine",             "speed_enhancing"),
    ("acufine",             "speed_enhancing"),
    ("ethol ufg",           "speed_enhancing"),
    ("sprint system",       "speed_enhancing"),
    ("divided d-76",        "speed_enhancing"),
    ("divided d76",         "speed_enhancing"),
    ("two bath",            "speed_enhancing"),
    ("2-bath",              "speed_enhancing"),
    ("two-bath",            "speed_enhancing"),

    # ── Compensating / high-acutance ─────────────────────────────────────────
    ("rodinal",             "compensating"),
    ("r09 one shot",        "compensating"),
    ("r09",                 "compensating"),
    ("r-09",                "compensating"),
    ("adonal",              "compensating"),
    ("agfa 8",              "compensating"),
    ("beutler",             "compensating"),
    ("windisch",            "compensating"),
    ("spur dokupan",        "compensating"),
    ("technidol",           "compensating"),
    ("parodinal",           "compensating"),    # also in alternative above; first hit wins
    ("blazinal",            "compensating"),

    # ── General purpose (last, broad matches) ────────────────────────────────
    ("d-76",                "general_purpose"),
    ("d76",                 "general_purpose"),
    ("id-11",               "general_purpose"),
    ("id11",                "general_purpose"),
    ("hc-110",              "general_purpose"),
    ("hc110",               "general_purpose"),
    ("ilfosol 3",           "general_purpose"),
    ("ilfosol s",           "general_purpose"),
    ("ilfosol",             "general_purpose"),
    ("ilford lc29",         "general_purpose"),
    ("lc29",                "general_purpose"),
    ("ilfotec",             "general_purpose"),
    ("kodak hc-110",        "general_purpose"),
    ("d-19",                "general_purpose"),
    ("d19",                 "general_purpose"),
    ("d-23",                "general_purpose"),
    ("d23",                 "general_purpose"),
    ("d-25",                "general_purpose"),
    ("d25",                 "general_purpose"),
    ("d-72",                "general_purpose"),
    ("fomadon",             "general_purpose"),
    ("fomadon r09",         "general_purpose"),   # note: more specific above
    ("fomadon lqn",         "general_purpose"),
    ("tetenal ultrafin plus", "general_purpose"),
    ("tetenal ultrafin",    "general_purpose"),
    ("rollei rhs",          "general_purpose"),
    ("rollei rpx-d",        "general_purpose"),
    ("studional",           "general_purpose"),
    ("novogam",             "general_purpose"),
    ("spur universal",      "general_purpose"),
    ("adox fx-39",          "general_purpose"),
    ("adox adonal",         "general_purpose"),
    ("foma fomadon",        "general_purpose"),
]

DEVELOPER_TYPE_LABELS: list[str] = [
    "general_purpose",
    "fine_grain",
    "speed_enhancing",
    "compensating",
    "tabular_grain",
    "staining",
    "alternative",
]
DEVELOPER_TYPE_CODES: dict[str, int] = {
    lbl: i for i, lbl in enumerate(DEVELOPER_TYPE_LABELS)
}


def _classify_developer(name: str) -> str:
    """Return developer type label for a given developer name string."""
    if not isinstance(name, str):
        return "general_purpose"
    lower = name.lower()
    for pattern, label in DEVELOPER_TYPE_RULES:
        if pattern in lower:
            return label
    return "general_purpose"


class DeveloperMetadataBuilder:
    """
    Adds one column to a DataFrame that has a 'Developer' column:

        developer_type_code  – integer-encoded developer type

    No fit() step required – all mappings are deterministic reference data.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a NEW DataFrame (same index) with the developer_type_code column.
        Does not modify the input.
        """
        out = pd.DataFrame(index=df.index)

        dev_type = df["Developer"].map(_classify_developer)
        out["developer_type_code"] = dev_type.map(DEVELOPER_TYPE_CODES).fillna(
            DEVELOPER_TYPE_CODES["general_purpose"]
        ).astype(int)

        return out

    @staticmethod
    def decode(code: int) -> str:
        if 0 <= code < len(DEVELOPER_TYPE_LABELS):
            return DEVELOPER_TYPE_LABELS[code]
        return "unknown"
