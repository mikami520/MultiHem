import csv
from collections import Counter

def check_id_length_and_duplicates(csv_path):
    # Read the IDs from the CSV
    ids = []
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            ids.append(row["ID"])  # Collect the IDs

    # Check the length of each ID
    id_lengths = {id_: len(id_) for id_ in ids}
    unique_lengths = set(id_lengths.values())
    print(f"Unique ID lengths: {unique_lengths}")

    if len(unique_lengths) > 1:
        print("Inconsistent ID lengths found:")
        for id_, length in id_lengths.items():
            if length != next(iter(unique_lengths)):  # Compare length with the first unique length
                print(f"ID: {id_}, Length: {length}")
    else:
        print("All IDs have consistent lengths.")

    # Check for duplicate IDs
    id_counts = Counter(ids)
    duplicates = {id_: count for id_, count in id_counts.items() if count > 1}

    if duplicates:
        print(f"Duplicate IDs found ({len(duplicates)}):")
        for id_, count in duplicates.items():
            print(f"ID: {id_}, Count: {count}")
    else:
        print("No duplicate IDs found.")

    # Save duplicates to a file
    duplicate_file_path = "duplicate_ids.txt"
    with open(duplicate_file_path, "w") as f:
        for id_, count in duplicates.items():
            f.write(f"{id_}: {count}\n")
    print(f"Duplicate IDs saved to {duplicate_file_path}")

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "KaggleProject/final_merged.csv"

    # Run the check
    check_id_length_and_duplicates(csv_path)
