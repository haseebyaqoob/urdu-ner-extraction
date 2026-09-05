

# ── Confirmed wrong DAY tokens (substring bug — contain "دن" inside them) ──
WRONG_DAY_TOKENS = {
    'دنیا', 'آمدن', 'المبارک', 'بدنام', 'جون', 'آمدنی',
    'دنیش', 'گردن', 'چند', 'بدنامی', 'عدنان', 'جامعہ',
    'خریدنے', 'پیرو', 'دردناک', '5دن', '2', '53', '49',
    'برتھ', 'ڈے', 'پیرا', '15', '12', 'ہے(دنیا', 'کندن',
    'مدنظر', 'دیدنی', 'ہےدنیا', 'پیروں', 'مارچ', 'رات',
}

# ── Label mapping: ZUNERA label → your unified schema label ──
# Anything not listed here becomes O (dropped from schema)
LABEL_MAP = {
    # Core mappings
    "PER"       : "PERSON",
    "LOC"       : "LOCATION",
    "ORG"       : "ORGANIZATION",
    "PLACE"     : "PLACE",
    "JOBTITLE"  : "JOBTITLE",
    "DATE"      : "DATE",
    "TIME"      : "TIME",
    "DAY"       : "DAY",
    "EVENT"     : "EVENT",
    "LANGUAGE"  : "LANGUAGE",
    "CARDINAL"  : "NUMBER",

    # Drop these (not in your 11-label schema) → O
    # PRODUCT, FOOD, ORDINAL, GPE, COLOR, MONEY,
    # QUANTITY, DISEASE, NORP, PERCENT, WORKOFART, LAW
    # (anything not in LABEL_MAP gets O automatically)
}

# ── Counters ──
day_wrong_fixed  = 0
label_dropped    = 0
label_kept       = 0
total_tokens     = 0

with open(INPUT, encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.rstrip("\n")

        # Blank line = sentence boundary
        if line.strip() == "":
            fout.write("\n")
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            # Shouldn't happen after step1, but safety fallback
            fout.write(f"{line.strip()}\tO\n")
            continue

        token = parts[0].strip()
        label = parts[1].strip()
        total_tokens += 1

        # ── Parse BIO prefix ──
        if label == "O":
            fout.write(f"{token}\tO\n")
            label_kept += 1
            continue

        bio_prefix = ""
        core_label = label

        if label.startswith("B-") or label.startswith("I-"):
            bio_prefix = label[:2]   # "B-" or "I-"
            core_label = label[2:]   # e.g. "DAY", "PER", "CARDINAL"

        # ── DAY cleanup: remove confirmed-wrong tokens ──
        if core_label == "DAY" and token in WRONG_DAY_TOKENS:
            fout.write(f"{token}\tO\n")
            day_wrong_fixed += 1
            continue

        # ── Map to unified schema ──
        if core_label in LABEL_MAP:
            new_label = bio_prefix + LABEL_MAP[core_label]
            fout.write(f"{token}\t{new_label}\n")
            label_kept += 1
        else:
            # Label not in schema → drop to O
            fout.write(f"{token}\tO\n")
            label_dropped += 1

print("Done.")
print(f"  Total tokens processed          : {total_tokens:,}")
print(f"  Wrong DAY tags fixed → O        : {day_wrong_fixed:,}")
print(f"  Labels dropped (not in schema)  : {label_dropped:,}")
print(f"  Labels kept (mapped to schema)  : {label_kept:,}")
print(f"  Output written to               : {OUTPUT}")
