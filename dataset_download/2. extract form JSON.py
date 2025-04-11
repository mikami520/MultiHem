import os
import shutil
import jsonlines

def extract_slices_from_jsonl(jsonl_file, dcm_directory, output_directory):
    """
    Extract the specified .dcm slices for each scan listed in the JSONL file.
    """
    with jsonlines.open(jsonl_file, 'r') as reader:
        for item in reader:
            scan_name = item['scan_']
            slice_list = item['slice_list']

            # Create an output folder for this scan if it doesn't exist
            scan_output_dir = os.path.join(output_directory, scan_name)
            if not os.path.exists(scan_output_dir):
                os.makedirs(scan_output_dir)

            # Copy each slice in the slice_list to the scan's output directory
            for slice_id in slice_list:
                source_file = os.path.join(dcm_directory, f"ID_{slice_id}.dcm")
                target_file = os.path.join(scan_output_dir, f"ID_{slice_id}.dcm")
                if os.path.exists(source_file):
                    shutil.copyfile(source_file, target_file)
                else:
                    print(f"WARNING: Missing slice ID_{slice_id}.dcm for scan {scan_name}")

if __name__ == "__main__":
    # Paths
    jsonl_file = r"KaggleProject\annotion_file_info.jsonl"  # JSONL file with slice info
    dcm_directory = r"KaggleProject\rsna_data\rsna-intracranial-hemorrhage-detection\stage_2_train"  # Source DICOM directory
    output_directory = r"KaggleProject\rsna_data2"  # New output folder for this approach

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the slices
    extract_slices_from_jsonl(jsonl_file, dcm_directory, output_directory)

    print(f"Slices have been extracted to: {output_directory}")

import SimpleITK as sitk

def dcm2nii(filepath, target_filepath):
    """
    Convert a series of DICOM files in a directory to a single NIfTI file.
    """
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    if not series_id:
        print(f"WARNING: No series found in {filepath}. Skipping.")
        return
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    images = series_reader.Execute()
    sitk.WriteImage(images, target_filepath)
    print(f"Converted {filepath} to {target_filepath}")

def convert_dcm_to_nifti(input_directory, output_directory):
    """
    Convert all DICOM files grouped in subdirectories under the input directory into NIfTI files.
    """
    # Iterate through all scan folders in the input directory
    for scan_name in os.listdir(input_directory):
        scan_folder = os.path.join(input_directory, scan_name)
        if not os.path.isdir(scan_folder):  # Ensure it's a directory
            continue

        # Output NIfTI file path
        nifti_output_path = os.path.join(output_directory, f"{scan_name}.nii.gz")

        # Convert DICOM files in the scan folder to a NIfTI file
        dcm2nii(scan_folder, nifti_output_path)

if __name__ == "__main__":
    # Paths
    input_directory = r"KaggleProject\rsna_data2"  # Extracted DICOM files directory
    output_directory = r"KaggleProject\nifti_data2"  # Directory to save NIfTI files

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Convert DICOM to NIfTI
    convert_dcm_to_nifti(input_directory, output_directory)

    print(f"All NIfTI files have been saved to: {output_directory}")
