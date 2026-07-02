#!/usr/bin/env python3
"""Fix mojibake in shl_product_catalog.json, then regenerate embeddings."""
import json
from pathlib import Path

catalog_path = Path("data/shl_product_catalog.json")
with catalog_path.open(encoding="utf-8") as f:
    cat = json.load(f)

# Map of bad byte-sequences (as Python unicode) to correct chars
EN_DASH = "\u2013"   # –
DEGREE  = "\u00b0"   # °

def fix_str(s: str) -> str:
    # â€" is the latin1-as-utf8 mojibake for the en dash –
    # The raw bytes 0xE2 0x80 0x93 decoded as latin1 give: â\x80\x93
    # but many scrapers encode it further as the 3-char sequence â + euro + right-double-quote
    s = s.replace("\u00e2\u20ac\u2013", EN_DASH)   # â€" (correct unicode codepoints)
    s = s.replace("\u00e2\u20ac\u201c", EN_DASH)   # â€" variant
    # Degree sign: Â° (C3 82 C2 B0 double-encoded)
    s = s.replace("\u00c2\u00b0", DEGREE)
    return s

fixed_items = []
for item in cat:
    original_name = item.get("name", "")
    new_name = fix_str(original_name)
    if new_name != original_name:
        fixed_items.append(f"  {original_name!r} -> {new_name!r}")
        item["name"] = new_name
    if "description" in item:
        item["description"] = fix_str(item["description"])

print(f"Fixed {len(fixed_items)} item names:")
for line in fixed_items:
    print(line)

with catalog_path.open("w", encoding="utf-8") as f:
    json.dump(cat, f, indent=2, ensure_ascii=False)

print(f"Saved {len(cat)} items to {catalog_path}")
