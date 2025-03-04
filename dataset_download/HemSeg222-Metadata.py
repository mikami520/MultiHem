import json
import pandas as pd
import re

# -------------------------------------------------------------
# STEP 1: Convert the JSONL file (HemSeg222) into a CSV file
#         "HemSeg222-slice-list.csv" where each row corresponds
#         to one slice_list ID along with its associated patient ID 
#         and scan_.
# -------------------------------------------------------------

# Path to your JSONL file (one JSON object per line)
hemseg_jsonl_file = "annotion_file_info.jsonl" 
hemseg_output_csv = "HemSeg222-slice-list.csv"

# Collect rows for the CSV
hemseg_rows = []

with open(hemseg_jsonl_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # Parse JSON record
        record = json.loads(line)
        patient = record.get("patient")
        scan = record.get("scan_")
        slice_list = record.get("slice_list", [])
        
        # Create rows for each slice_list ID
        for slice_id in slice_list:
            hemseg_rows.append({
                "ID": slice_id,
                "patient": patient,
                "scan": scan
            })

# Create a DataFrame and save it to CSV
df_hemseg = pd.DataFrame(hemseg_rows)
df_hemseg.to_csv(hemseg_output_csv, index=False)
print(f"HemSeg222 slice list saved to '{hemseg_output_csv}'.")

# Validate the number of unique patients
unique_patients = df_hemseg['patient'].nunique()
if unique_patients == 222:
    print(f"Validation successful: {unique_patients} unique patients found.")
else:
    raise ValueError(f"Validation failed: Expected 222 unique patients, but found {unique_patients}.")

# -------------------------------------------------------------
# STEP 2: Flatten the RSNA CSV file ("stage_2_train.csv")
#
#   The input CSV file has two columns:
#       ID,Label
#
#   Example rows (for one study):
#       ID_12cadc6af_epidural,0
#       ID_12cadc6af_intraparenchymal,0
#       ...
#
#   We will flatten it so that each unique base ID (after removing the 
#   "ID_" prefix) becomes one row with 7 columns:
#       ID, epidural, intraparenchymal, intraventricular,
#       subarachnoid, subdural, any
#
#   Also, the output ID column will not contain the initial "ID_".
#   The output will be saved as "RSNA-slice-list.csv".
# -------------------------------------------------------------

rsna_input_csv = "stage_2_train.csv" 
rsna_output_csv = "RSNA-slice-list.csv"

# Load the RSNA CSV file.
df_rsna = pd.read_csv(rsna_input_csv)

# Extract participant ID and hemorrhage type by splitting the 'ID' column.
df_rsna['Participant_ID'] = df_rsna['ID'].str.split('_').str[1]
df_rsna['Hemorrhage_Type'] = df_rsna['ID'].str.split('_').str[2]

# Pivot the table so that each unique participant ID becomes one row,
# with columns for each hemorrhage type.
# In case of duplicates, we use 'max' as the aggregation (so if any duplicate is 1, result is 1)
df_rsna_flat = df_rsna.pivot_table(
    index="Participant_ID",
    columns="Hemorrhage_Type",
    values="Label",
    aggfunc="max"
).reset_index()

# Rename the "Participant_ID" column to "ID" (so that it no longer contains "ID_" or similar prefix)
df_rsna_flat.rename(columns={"Participant_ID": "ID"}, inplace=True)

# Reorder the columns to: ID, epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
expected_labels = ['epidural', 'intraparenchymal', 'intraventricular',
                   'subarachnoid', 'subdural', 'any']
final_columns = ["ID"] + [label for label in expected_labels if label in df_rsna_flat.columns]
df_rsna_flat = df_rsna_flat[final_columns]

# Save the flattened RSNA CSV.
df_rsna_flat.to_csv(rsna_output_csv, index=False)
print(f"Flattened RSNA CSV saved to '{rsna_output_csv}'.")


# -------------------------------------------------------------
# STEP 3: Cross-reference the two datasets
#
#   We now have:
#     • "RSNA-slice-list.csv" where each row is a unique slice ID (no more "ID_" prefix)
#       along with 6 binary label columns.
#
#     • "HemSeg222-slice-list.csv": each row is one slice_list ID (as given in the JSONL), 
#       with the associated patient and scan information.
#
#   We want to merge these two files using an inner join on the ID column.
#   Rows in RSNA-slice-list that do not appear in HemSeg222-slice-list will be dropped.
#
#   The final merged CSV is saved as "final_merged.csv".
# -------------------------------------------------------------

# Load the two CSV files.
df_hemseg = pd.read_csv(hemseg_output_csv)
df_rsna_flat = pd.read_csv(rsna_output_csv)

# Merge on column "ID". Using an inner join ensures we only keep rows that are in HemSeg222.
df_merged = pd.merge(df_rsna_flat, df_hemseg, on="ID", how="inner")

final_output_csv = "final_merged.csv"
df_merged.to_csv(final_output_csv, index=False)
print(f"Final merged CSV saved to '{final_output_csv}'.")

# -------------------------------------------------------------
# STEP 4: Summary statistics
import pandas as pd
# Load the final_merged.csv
input_csv = "final_merged.csv"
output_csv = "patient_analysis.csv"
df = pd.read_csv(input_csv)

# Count slice-level categories
df['Category'] = "Neither"
df.loc[(df['intraparenchymal'] == 1) & (df['intraventricular'] == 0), "Category"] = "Intraparenchymal Only"
df.loc[(df['intraparenchymal'] == 0) & (df['intraventricular'] == 1), "Category"] = "Intraventricular Only"
df.loc[(df['intraparenchymal'] == 1) & (df['intraventricular'] == 1), "Category"] = "Both"

slice_counts = df['Category'].value_counts()
print("Slice-Level Counts:")
print(slice_counts)

'''
Category
Neither                  5665
Intraventricular Only     899
Intraparenchymal Only     795
'''

# Count patient-level categories
total_patients = df['patient'].nunique()
print(f"Total number of unique patients: {total_patients}")

df_patients = df.groupby("patient")[["intraparenchymal", "intraventricular"]].max().reset_index()
# kepp the scan ID for each patient 
df_scan_id = df.groupby("patient")["scan"].first().reset_index()
df_patients = pd.merge(df_patients, df_scan_id, on="patient",how="left")
# Add a new column "Category" to indicate the hemorrhage type(s) for each patient
df_patients['Category'] = "Neither"
df_patients.loc[(df_patients['intraparenchymal'] == 1) & (df_patients['intraventricular'] == 0), "Category"] = "Intraparenchymal Only"
df_patients.loc[(df_patients['intraparenchymal'] == 0) & (df_patients['intraventricular'] == 1), "Category"] = "Intraventricular Only"
df_patients.loc[(df_patients['intraparenchymal'] == 1) & (df_patients['intraventricular'] == 1), "Category"] = "Both"

patient_counts = df_patients['Category'].value_counts()
print("Patient-Level Counts:")
print(patient_counts)

'''
Category
Intraparenchymal Only    108
Intraventricular Only    101
'''


# Save the patient-level data to a CSV
df_patients.to_csv(output_csv, index=False)
print(f"Patient analysis saved to '{output_csv}'.")


