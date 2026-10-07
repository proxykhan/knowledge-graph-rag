"""The corpus: US-listed semiconductor companies that file 10-Ks.

Chosen because their filings name each other as customers, suppliers, competitors and
acquisition targets, which gives the graph real multi-hop paths (e.g. equipment maker
-> supplies -> chipmaker -> competes with -> chipmaker).
Foreign filers like TSMC and ASML file 20-F instead; they still appear as entities
because the 10-Ks below mention them.
"""

TICKERS = [
    # Chip designers / makers
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MRVL", "ON",
    "MCHP", "NXPI", "SWKS", "QRVO", "WDC",
    # Equipment, materials, packaging (suppliers to the above)
    "AMAT", "LRCX", "KLAC", "ENTG", "AMKR",
]

# Clean legal names. SEC titles look like "APPLIED MATERIALS INC /DE", and the LLM copies
# whatever name we give it for "we", so it must be the canonical one.
COMPANY_NAMES = {
    "NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices, Inc.", "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Incorporated", "AVGO": "Broadcom Inc.", "MU": "Micron Technology, Inc.",
    "TXN": "Texas Instruments Incorporated", "ADI": "Analog Devices, Inc.",
    "MRVL": "Marvell Technology, Inc.", "ON": "ON Semiconductor Corporation",
    "MCHP": "Microchip Technology Incorporated", "NXPI": "NXP Semiconductors N.V.",
    "SWKS": "Skyworks Solutions, Inc.", "QRVO": "Qorvo, Inc.", "WDC": "Western Digital Corporation",
    "AMAT": "Applied Materials, Inc.", "LRCX": "Lam Research Corporation",
    "KLAC": "KLA Corporation", "ENTG": "Entegris, Inc.", "AMKR": "Amkor Technology, Inc.",
}

# Small subset for development runs, so extraction costs stay low while iterating.
DEV_TICKERS = ["NVDA", "AMD", "INTC", "AMAT", "MU"]

# Filers whose 10-K body has no "Item X." headings. Intel uses its own layout plus a
# page-number cross-reference index, so sections are located by start/end line markers.
# Markers are tied to one filing year; re-check them when a new 10-K is downloaded.
SECTION_MARKERS: dict[str, dict[str, tuple[str, str]]] = {
    "INTC": {
        "1": (r"^Our Priorities$", r"^Significant Events and Trends Impacting Results$"),
        "1A": (r"^The following summarizes what we believe to be the material factors",
               r"^Information About Our Executive Officers$"),
        "7": (r"^Significant Events and Trends Impacting Results$", r"^Properties$"),
    },
}
