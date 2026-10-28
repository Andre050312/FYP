"""Register ADFTDRSLoader in data_provider/data_loader.py exactly once."""

from pathlib import Path


path = Path(r"C:\Users\ROG\Desktop\LEAD\data_provider\data_loader.py")
text = path.read_text(encoding="utf-8")

new_import = "from data_provider.dataset_loader.adftd_rs_loader import ADFTDRSLoader\n"
anchor_import = "from data_provider.dataset_loader.adftd_loader import ADFTDLoader\n"
if new_import not in text:
    if anchor_import not in text:
        raise RuntimeError("Could not find ADFTDLoader import anchor")
    text = text.replace(anchor_import, anchor_import + new_import, 1)

old_mapping = "    'ADFTD-RS': ADFTDLoader,  # resting-state subset of ADFTD"
new_mapping = "    'ADFTD-RS': ADFTDRSLoader,  # OpenNeuro ds004504 resting-state data"
if old_mapping in text:
    text = text.replace(old_mapping, new_mapping, 1)
elif new_mapping not in text:
    raise RuntimeError("Could not find ADFTD-RS mapping anchor")

path.write_text(text, encoding="utf-8")
print(f"Registered ADFTDRSLoader in {path}")
