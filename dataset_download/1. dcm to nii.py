import os
import shutil
import sys
import SimpleITK as sitk
import jsonlines

def dcm2nii(filepath, target_filepath, log_file):
    """
    Convert a series of DICOM files in a directory to a single NIfTI file.
    """
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    if not series_id:
        log_message = f"WARNING: No series found in {filepath}. Skipping.\n"
        print(log_message)
        with open(log_file, "a") as log:
            log.write(log_message)
        return
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    images = series_reader.Execute()
    sitk.WriteImage(images, target_filepath)
    log_message = f"Converted {filepath} to {target_filepath}\n"
    print(log_message)
    with open(log_file, "a") as log:
        log.write(log_message)

def convert_dcm_to_nifti(jsonl_file, dcm_directory, output_directory, log_file):
    """
    Convert DICOM slices to NIfTI files based on the information in the JSONL file.
    """
    with jsonlines.open(jsonl_file, 'r') as reader:
        for item in reader:
            scan_name = item['scan_']
            slice_list = item['slice_list']

            # Create a temporary folder to store the DICOM files for this scan
            temp_scan_dir = os.path.join(dcm_directory, scan_name)
            if not os.path.exists(temp_scan_dir):
                os.makedirs(temp_scan_dir)

            # Copy the required DICOM slices to the temporary folder
            for slice_id in slice_list:
                source_file = os.path.join(dcm_directory, f"ID_{slice_id}.dcm")
                target_file = os.path.join(temp_scan_dir, f"ID_{slice_id}.dcm")
                if os.path.exists(source_file):
                    shutil.copyfile(source_file, target_file)
                else:
                    log_message = f"WARNING: Missing slice {slice_id} for scan {scan_name}. Skipping.\n"
                    print(log_message)
                    with open(log_file, "a") as log:
                        log.write(log_message)

            # Convert the DICOM files in the temporary folder to a NIfTI file
            nifti_output_path = os.path.join(output_directory, f"{scan_name}.nii.gz")
            dcm2nii(temp_scan_dir, nifti_output_path, log_file)

            # Clean up the temporary folder
            shutil.rmtree(temp_scan_dir)

if __name__ == "__main__":
    # Paths
    jsonl_file = r"KaggleProject\annotion_file_info.jsonl"
    dcm_directory = r"KaggleProject\rsna_data\rsna-intracranial-hemorrhage-detection\stage_2_train"
    output_directory = r"KaggleProject\nifti_data"
    log_file = "conversion_logs.txt"

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Clear the log file if it exists
    if os.path.exists(log_file):
        with open(log_file, "w") as log:
            log.write("Log for NIfTI Conversion Process\n")

    # Convert DICOM to NIfTI and log the process
    convert_dcm_to_nifti(jsonl_file, dcm_directory, output_directory, log_file)
