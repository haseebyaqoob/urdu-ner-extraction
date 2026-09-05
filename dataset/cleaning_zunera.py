

INPUT  = r"file path to your input file"  # Replace with the actual path to your input file
OUTPUT = r"file path to your output file"  # Replace with the actual path to your output file

fixed   = 0
missing = 0
total   = 0

with open(INPUT, encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for raw_line in fin:
        line = raw_line.rstrip("\r\n")

        # Skip header
        if line == "text\tlabels":
            continue

        # Blank line = sentence boundary — keep it
        if line.strip() == "":
            fout.write("\n")
            continue

        total += 1

        # Case 1: already correct (tab-separated)
        if "\t" in line:
            parts = line.split("\t")
            token = parts[0].strip()
            label = parts[-1].strip()
            fout.write(f"{token}\t{label}\n")

        # Case 2: space-separated (token SPACE label)
        elif " " in line.strip():
            parts = line.strip().rsplit(" ", 1)   # split on LAST space
            token = parts[0].strip()
            label = parts[1].strip()
            fout.write(f"{token}\t{label}\n")
            fixed += 1

        # Case 3: token with no label at all → assign O
        else:
            token = line.strip()
            fout.write(f"{token}\tO\n")
            missing += 1

print(f"Done.")
print(f"  Total token lines processed : {total:,}")
print(f"  Space-delimited lines fixed : {fixed:,}")
print(f"  Missing-label lines → O    : {missing:,}")
print(f"  Output written to          : {OUTPUT}")
