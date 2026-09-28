import json
import glob
from collections import Counter

files = sorted(glob.glob("obis_fish_*.json"))

records = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        records.extend(json.load(f))

print(f"Raw records loaded: {len(records)}")

ids = []
for r in records:
    occurrence_id = r.get("occurrenceID") or r.get("id")
    if occurrence_id:
        ids.append(occurrence_id)

counts = Counter(ids)

duplicate_records = sum(count - 1 for count in counts.values() if count > 1)
duplicate_ids = sum(1 for count in counts.values() if count > 1)

unique_records = len(counts)

print(f"Records with occurrence IDs: {len(ids)}")
print(f"Unique occurrence IDs: {unique_records}")
print(f"Duplicate occurrence IDs: {duplicate_ids}")
print(f"Duplicate records beyond first copy: {duplicate_records}")

print("\nTop duplicate IDs:")
for occurrence_id, count in counts.most_common(20):
    if count > 1:
        print(f"{count}x  {occurrence_id}")