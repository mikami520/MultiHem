import pandas as pd

# Load the CSV file
file_path = "stage_2_train.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Extract participant ID and hemorrhage type
df['Participant_ID'] = df['ID'].str.split('_').str[1]
df['Hemorrhage_Type'] = df['ID'].str.split('_').str[2]

# Filter participants with IPH (intraparenchymal) or IVH (intraventricular) labeled as 1
filtered_df = df[(df['Hemorrhage_Type'].isin(['intraparenchymal', 'intraventricular'])) & (df['Label'] == 1)]
print(f"Number of rows after filtering: {len(filtered_df)}")
                                                                                           
# Create a summary DataFrame indicating IPH or IVH for each participant
summary_df = filtered_df.pivot_table(
    index='Participant_ID',
    columns='Hemorrhage_Type',
    values='Label',
    aggfunc='max',  # Maximum value to represent presence (1) of each hemorrhage type
    fill_value=0    # Replace NaN with 0
).reset_index()

# Save the summary to a CSV
output_path = "selected_participants_with_IPH_IVH.csv"  # Output file name
summary_df.to_csv(output_path, index=False)
print(f"Summary saved to {output_path}")

# Count total number of participants for each group
both_count = summary_df[(summary_df['intraparenchymal'] == 1) & (summary_df['intraventricular'] == 1)].shape[0]  # Both IPH and IVH
iph_only_count = summary_df[(summary_df['intraparenchymal'] == 1) & (summary_df['intraventricular'] == 0)].shape[0]  # Only IPH
ivh_only_count = summary_df[(summary_df['intraparenchymal'] == 0) & (summary_df['intraventricular'] == 1)].shape[0]  # Only IVH
neither_count = summary_df[(summary_df['intraparenchymal'] == 0) & (summary_df['intraventricular'] == 0)].shape[0]  # Neither

# Print the updated summary
print("Summary:")
print(f"Participants with only IPH (intraparenchymal hemorrhage): {iph_only_count}")
print(f"Participants with only IVH (intraventricular hemorrhage): {ivh_only_count}")
print(f"Participants with both IPH and IVH: {both_count}")
print(f"Participants with neither IPH nor IVH: {neither_count}")

'''
Summary:
Participants with only IPH (intraparenchymal hemorrhage): 5955
Participants with only IVH (intraventricular hemorrhage): 3621
Participants with both IPH and IVH: 2504
Participants with neither IPH nor IVH: 0
'''
