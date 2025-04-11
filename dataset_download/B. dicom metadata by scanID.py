import pandas as pd
import json
from collections import defaultdict

# Input and output file names
input_file = "dicom_metadata_with_labels.csv"
patient_analysis_csv = "patient_hemorrhage_analysis.csv"
patient_slices_jsonl = "patient_slices.jsonl"

# Load the input data
df = pd.read_csv(input_file)

# Prepare data for CSV: Group by PatientID and StudyInstanceUID
grouped = df.groupby(['PatientID', 'StudyInstanceUID'])

# Initialize a list for processed CSV rows and a list for JSONL objects
csv_rows = []
jsonl_objects = []

# Dictionary to keep track of duplicate PatientIDs
patient_id_counts = defaultdict(int)

# Initialize counters
total_slices = 0

for (patient_id, study_instance_uid), group in grouped:
    # Count duplicate PatientIDs and handle unique IDs
    patient_id_counts[patient_id] += 1
    if patient_id_counts[patient_id] > 1:
        new_patient_id = f"{patient_id}_dup{patient_id_counts[patient_id]}"
    else:
        new_patient_id = patient_id

    # Hemorrhage labels for the entire scan (any slice labeled 1 results in a 1 for the scan)
    hemorrhage_labels = {
        "Epidural": group['Epidural'].max(),
        "Intraparenchymal": group['Intraparenchymal'].max(),
        "Intraventricular": group['Intraventricular'].max(),
        "Subarachnoid": group['Subarachnoid'].max(),
        "Subdural": group['Subdural'].max(),
        "Any": group['Any'].max()
    }

    # Add a row for the CSV
    csv_row = {
        "PatientID": new_patient_id,
        "StudyInstanceUID": study_instance_uid,
        **hemorrhage_labels
    }
    csv_rows.append(csv_row)

    # Extract only the slice IDs without the prefix "ID_"
    slice_list = [filename.split('/')[-1].split('.dcm')[0].replace("ID_", "") for filename in group['DICOM_FileName']]
    total_slices += len(slice_list)

    # Add an object for the JSONL file (excluding hemorrhage labels)
    jsonl_object = {
        "patient": new_patient_id,
        "scan": study_instance_uid,
        "slice_list": slice_list
    }
    jsonl_objects.append(jsonl_object)

# Create and save the CSV file
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(patient_analysis_csv, index=False)
print(f"Patient-level CSV file saved to {patient_analysis_csv}")

# Create and save the JSONL file
with open(patient_slices_jsonl, 'w') as jsonl_file:
    for obj in jsonl_objects:
        jsonl_file.write(json.dumps(obj) + '\n')
print(f"Patient-level JSONL file saved to {patient_slices_jsonl}")

# Report totals
total_patients = len(patient_id_counts)
total_scans = len(csv_rows)
print(f"Total number of patients: {total_patients}")
print(f"Total number of scans: {total_scans}")
print(f"Total number of slices: {total_slices}")

'''
Patient-level CSV file saved to patient_hemorrhage_analysis.csv
Patient-level JSONL file saved to patient_slices.jsonl
Total number of patients: 18938
Total number of scans: 21744
Total number of slices: 752803
'''