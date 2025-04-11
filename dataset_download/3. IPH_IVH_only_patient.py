import pandas as pd

# Load the patient-level analysis CSV
df = pd.read_csv("patient_hemorrhage_analysis.csv")

# Define filters for "IPH only" and "IVH only".
# For IPH only: Intraparenchymal = 1 and all others (Intraventricular, Epidural, Subarachnoid, Subdural) equal 0.
# For IVH only: Intraventricular = 1 and all others (Intraparenchymal, Epidural, Subarachnoid, Subdural) equal 0.

iph_only = (
    (df["Intraparenchymal"] == 1) &
    (df["Intraventricular"] == 0) &
    (df["Epidural"] == 0) &
    (df["Subarachnoid"] == 0) &
    (df["Subdural"] == 0)
)

ivh_only = (
    (df["Intraventricular"] == 1) &
    (df["Intraparenchymal"] == 0) &
    (df["Epidural"] == 0) &
    (df["Subarachnoid"] == 0) &
    (df["Subdural"] == 0)
)

# Combine the two filters using OR. This selects patients with either IPH only or IVH only.
df_filtered = df[iph_only | ivh_only]

# Write the filtered patients to CSV
df_filtered.to_csv("patient_IPH_or_IVH_only.csv", index=False)
print("Filtered patient (IPH only or IVH only, without other bleeding) CSV saved as patient_IPH_or_IVH_only.csv")

# 1508 scans in total #

# --------------------------------------------------
## Below are steps to convert dicoms to nifti files, excluding the patients in the HemSeg-200 dataset.
# --------------------------------------------------
import os
import csv
import jsonlines
import zipfile
import SimpleITK as sitk

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def load_excluded_scans(jsonl_path):
    """
    Reads 'annotion_file_info.jsonl' to create a set of scan IDs to exclude.
    """
    exclude_scans = set()
    with jsonlines.open(jsonl_path, 'r') as reader:
        for record in reader:
            scan_id = record.get("scan_")
            if scan_id:
                exclude_scans.add(scan_id)
    return exclude_scans


# Load the excluded scan IDs from the JSONL file
jsonl_path = "annotion_file_info.jsonl"
excluded_scans = load_excluded_scans(jsonl_path)

# Ensure excluded scan IDs are consistent with StudyInstanceUID by adding the "ID_" prefix
excluded_scans_with_prefix = {f"ID_{scan_id}" for scan_id in excluded_scans}

# Add a new column 'seg' to indicate scan presence in the JSONL file
df_filtered = pd.read_csv("patient_IPH_or_IVH_only.csv")
df_filtered["seg"] = df_filtered["StudyInstanceUID"].apply(lambda x: 1 if x in excluded_scans_with_prefix else 0)

# Save the updated DataFrame to a new CSV file
df_filtered.to_csv("patient_IPH_or_IVH_only_with_seg.csv", index=False)
print("Updated CSV with 'seg' column saved as patient_IPH_or_IVH_only_with_seg.csv")

def load_valid_scans_from_csv(csv_path):
    """
    Reads 'patient_IPH_or_IVH_only.csv' and returns a set of valid StudyInstanceUID (scan IDs).
    Assumes the file has a header and the second column is 'StudyInstanceUID'.
    """
    valid_scans = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)  # Use DictReader to handle CSV headers
        for row in reader:
            scan_id = row.get("StudyInstanceUID")
            if scan_id:
                valid_scans.add(scan_id)
    return valid_scans

def load_patient_records(jsonl_path, exclude_scans, valid_scans):
    """
    Reads 'patient_slices.jsonl' and returns records:
    - Excludes those with excluded scan IDs.
    - Includes only those with scan IDs in valid_scans.
    """
    records = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for record in reader:
            scan_id = record.get("scan")
            if not scan_id or scan_id in exclude_scans or scan_id not in valid_scans:
                continue
            records.append(record)
    return records

def extract_slices_from_patient_records(records, zip_path, output_directory):
    """
    Extracts DICOM files based on patient records.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        zip_contents = set(z.namelist())
        print(f"Zip file contains {len(zip_contents)} items.")
        
        for record in records:
            scan = record.get("scan")
            slice_list = record.get("slice_list", [])
            if not scan:
                print(f"WARNING: Record for patient {record.get('patient')} has no scan info. Skipping.")
                continue
            
            # Create folder for this scan if not exists
            scan_output_dir = os.path.join(output_directory, scan)
            if not os.path.exists(scan_output_dir):
                os.makedirs(scan_output_dir)
            
            # Copy each slice file from the zip
            for slice_id in slice_list:
                filename_in_zip = f"rsna-intracranial-hemorrhage-detection/stage_2_train/ID_{slice_id}.dcm"
                target_file = os.path.join(scan_output_dir, f"ID_{slice_id}.dcm")
                if filename_in_zip in zip_contents:
                    try:
                        data = z.read(filename_in_zip)
                        with open(target_file, 'wb') as f_out:
                            f_out.write(data)
                    except Exception as e:
                        print(f"ERROR: Could not extract {filename_in_zip}: {e}")
                else:
                    print(f"WARNING: {filename_in_zip} not found in the zip.")

def dcm2nii(input_folder, output_filepath):
    """
    Converts a folder of DICOM files into a single NIfTI file.
    """
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_folder)
    if not series_ids:
        print(f"WARNING: No DICOM series found in {input_folder}. Skipping conversion.")
        return False
    try:
        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder, series_ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files)
        image = reader.Execute()
        sitk.WriteImage(image, output_filepath)
        print(f"Converted {input_folder} to {output_filepath}")
        return True
    except Exception as e:
        print(f"ERROR converting {input_folder}: {e}")
        return False

def convert_extracted_dicoms_to_nifti(extraction_dir, nii_output_dir):
    """
    Converts all extracted DICOM folders to NIfTI format.
    """
    if not os.path.exists(nii_output_dir):
        os.makedirs(nii_output_dir)
    
    for scan in os.listdir(extraction_dir):
        scan_folder = os.path.join(extraction_dir, scan)
        if not os.path.isdir(scan_folder):
            continue
        output_nii = os.path.join(nii_output_dir, f"{scan}.nii.gz")
        dcm2nii(scan_folder, output_nii)

# --------------------------------------------------
# Main Processing Routine
# --------------------------------------------------
if __name__ == "__main__":
    # Paths to required files and directories
    zip_path = "rsna-intracranial-hemorrhage-detection.zip"
    exclude_jsonl = r"annotion_file_info.jsonl"
    patient_slices_jsonl = r"patient_slices.jsonl"
    slice_csv_path = r"patient_IPH_or_IVH_only.csv"
    extracted_dicoms_dir = r"rsna_data3"
    nifti_output_dir = r"nifti_data3"

    # Load exclusion list from 'annotion_file_info.jsonl'
    print("Loading exclusion list from", exclude_jsonl)
    exclude_scans = load_excluded_scans(exclude_jsonl)
    print(f"Excluding {len(exclude_scans)} scans from processing.")

    # Load valid scan IDs from 'patient_IPH_or_IVH_only.csv'
    print("Loading valid scan IDs from", slice_csv_path)
    valid_scans = load_valid_scans_from_csv(slice_csv_path)
    print(f"Loaded {len(valid_scans)} valid scan IDs for processing.")

    # Load patient slice records from 'patient_slices.jsonl'
    print("Loading patient slice records from", patient_slices_jsonl)
    patient_records = load_patient_records(patient_slices_jsonl, exclude_scans, valid_scans)
    print(f"Processing {len(patient_records)} patient records.")

    # Extract DICOM slices from the zip
    print("Extracting DICOM slices from zip...")
    extract_slices_from_patient_records(patient_records, zip_path, extracted_dicoms_dir)

    # Convert extracted DICOM series to NIfTI
    print("Converting extracted DICOM folders to NIfTI format...")
    convert_extracted_dicoms_to_nifti(extracted_dicoms_dir, nifti_output_dir)
    
    print("Processing completed:")
    print(f"  - Extracted DICOMs are saved in: {extracted_dicoms_dir}")
    print(f"  - NIfTI files are saved in: {nifti_output_dir}")
