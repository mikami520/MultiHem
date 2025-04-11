import zipfile
import pydicom
import pandas as pd
import csv
import json
from collections import defaultdict

# -------------------------------
# File paths and configuration
# -------------------------------
zip_path = "rsna-intracranial-hemorrhage-detection.zip"
output_csv = "dicom_metadata_with_labels.csv"
patient_analysis_csv = "patient_hemorrhage_analysis.csv"
patient_slices_jsonl = "patient_slices.jsonl"

# Set TEST_MODE = True to process only the first 100 DICOM files
TEST_MODE = False
max_files = 100 if TEST_MODE else None

# -----------------------------------------------
# Step 1: Build a lookup dictionary for labels
# -----------------------------------------------
# Instead of reading the labels from the zip file, we pull them directly
# from RSNA-slice-list.csv. This file contains rows with columns:
# ID, epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
labels_df = pd.read_csv("RSNA-slice-list.csv")

# Build a dictionary where the key is the slice ID (extracted from the "ID" column)
# and the value is a dictionary of hemorrhage labels.
labels_lookup = {}
for _, row in labels_df.iterrows():
    # The RSNA slice ID in the CSV is something like "000012eaf"
    key = str(row["ID"]).strip()
    labels_lookup[key] = {
        "Epidural": int(row["epidural"]),
        "Intraparenchymal": int(row["intraparenchymal"]),
        "Intraventricular": int(row["intraventricular"]),
        "Subarachnoid": int(row["subarachnoid"]),
        "Subdural": int(row["subdural"]),
        "Any": int(row["any"])
    }

# A dictionary to use as default if a slice is not found in the lookup
default_labels = {
    "Epidural": 0,
    "Intraparenchymal": 0,
    "Intraventricular": 0,
    "Subarachnoid": 0,
    "Subdural": 0,
    "Any": 0
}

# --------------------------------------------------
# Step 2: Process the DICOM files from the zip archive
# --------------------------------------------------
with zipfile.ZipFile(zip_path, 'r') as z:
    # Filter to get list of all DICOM files in stage_2_train folder
    dcm_files = [f for f in z.namelist() 
                 if f.startswith("rsna-intracranial-hemorrhage-detection/stage_2_train/") and f.endswith(".dcm")]
    
    # Counter for test mode. If in test mode, only process the first 'max_files' files.
    files_processed = 0

    # Prepare to write the slice-level CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        # Write the header for the slice-level CSV output
        writer.writerow([
            "DICOM_FileName", "SliceID", "PatientID", "StudyInstanceUID",
            "SeriesInstanceUID", "Epidural", "Intraparenchymal",
            "Intraventricular", "Subarachnoid", "Subdural", "Any"
        ])
    
        # Dictionary to track patient-level aggregation
        patient_data = defaultdict(lambda: {
            "scan": "",
            "slice_list": [],
            "Epidural": 0,
            "Intraparenchymal": 0,
            "Intraventricular": 0,
            "Subarachnoid": 0,
            "Subdural": 0,
            "Any": 0
        })
    
        for file_name in dcm_files:
            # If in test mode, only process up to max_files
            if TEST_MODE and files_processed >= max_files:
                break

            # A DICOM file name looks like:
            # "rsna-intracranial-hemorrhage-detection/stage_2_train/ID_000012eaf.dcm"
            # Extract the slice ID by removing the path and ".dcm"
            full_slice_id = file_name.split("/")[-1].replace(".dcm", "")  # e.g., "ID_000012eaf"
            # Remove the "ID_" prefix to match keys in labels_lookup (which are like "000012eaf")
            slice_id = full_slice_id.replace("ID_", "")
    
            # Read the DICOM file from the zip
            with z.open(file_name) as dicom_file:
                try:
                    # Read metadata (skip pixel data) to optimize runtime
                    dataset = pydicom.dcmread(dicom_file, stop_before_pixels=True)
    
                    # Extract DICOM metadata with a fallback of "Unknown"
                    patient_id = dataset.get("PatientID", "Unknown")
                    study_instance_uid = dataset.get("StudyInstanceUID", "Unknown")
                    series_instance_uid = dataset.get("SeriesInstanceUID", "Unknown")
    
                    # Pull hemorrhage labels directly from the lookup dictionary using slice_id
                    hemorrhage_types = labels_lookup.get(slice_id, default_labels)
    
                    # Write slice-level data to CSV
                    writer.writerow([
                        file_name, slice_id, patient_id, study_instance_uid,
                        series_instance_uid, hemorrhage_types["Epidural"],
                        hemorrhage_types["Intraparenchymal"],
                        hemorrhage_types["Intraventricular"],
                        hemorrhage_types["Subarachnoid"],
                        hemorrhage_types["Subdural"], hemorrhage_types["Any"]
                    ])
    
                    # Update patient-level aggregation if patient_id is valid
                    if patient_id != "Unknown":
                        # Use the study_instance_uid as a scan identifier (set only once per patient)
                        if not patient_data[patient_id]["scan"]:
                            patient_data[patient_id]["scan"] = study_instance_uid
                        # Append the slice id to this patient's slice list
                        patient_data[patient_id]["slice_list"].append(slice_id)
    
                        # For each hemorrhage type, if any slice is positive, mark the patient positive (1)
                        for key in hemorrhage_types:
                            if hemorrhage_types[key] == 1:
                                patient_data[patient_id][key] = 1
                                
                    files_processed += 1
    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

# ----------------------------------------------------
# Step 3: Output Patient-level Analysis (CSV & JSONL)
# ----------------------------------------------------
# Write patient-level hemorrhage analysis where each row contains:
# PatientID, StudyInstanceUID, and labels for each hemorrhage type.
with open(patient_analysis_csv, 'w', newline='', encoding='utf-8') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["PatientID", "StudyInstanceUID", "Epidural", "Intraparenchymal",
                     "Intraventricular", "Subarachnoid", "Subdural", "Any"])
    for patient_id, data in patient_data.items():
        writer.writerow([
            patient_id, data["scan"], data["Epidural"], data["Intraparenchymal"],
            data["Intraventricular"], data["Subarachnoid"], data["Subdural"],
            data["Any"]
        ])

# Write patient-associated slices as a JSONL file.
with open(patient_slices_jsonl, 'w', encoding='utf-8') as jsonl_file:
    for patient_id, data in patient_data.items():
        json_obj = {
            "patient": patient_id,
            "scan_": data["scan"],
            "slice_list": data["slice_list"]
        }
        jsonl_file.write(json.dumps(json_obj) + "\n")

# ----------------------------------------------------
# Final messages and test mode info
# ----------------------------------------------------
print("Slice-level metadata and labels saved to:", output_csv)
print("Patient-level analysis saved to:", patient_analysis_csv)
print("Patient-associated slices saved to JSONL:", patient_slices_jsonl)

if TEST_MODE:
    print("Test mode active: Processed first 100 DICOM files only; verify output before full run.")

# dicom_metadata_with_labels.csv has 752804 rows
# patient_hemorrhage_analysis.csv has 18938 patients, should be 21744 patients!!
