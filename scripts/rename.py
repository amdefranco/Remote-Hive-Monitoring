import csv

# Function to generate new rows for each WAV file
def expand_csv(input_csv, output_csv):
    # Open original CSV file for reading
    with open(input_csv, 'r') as input_file:
        reader = csv.DictReader(input_file)
        
        # Open new CSV file for writing
        with open(output_csv, 'w', newline='') as output_file:
            fieldnames = reader.fieldnames  # Get field names from original CSV
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()  # Write header to new CSV
            
            # Iterate over rows in original CSV
            for row in reader:
                # Extract information from the row
                filename = row['filename']
                target = row['target']
                
                # Generate new rows for each WAV file
                for i in range(6):  # Assuming 6 WAV files per sample
                    new_row = {
                        'filename': f"{filename}_{i}.wav",  # Modify filename for each WAV file
                        'target': target
                        # Add other columns if present in the original CSV
                    }
                    writer.writerow(new_row)  # Write new row to new CSV

# Example usage
input_csv = 'all_data_updated.csv'
output_csv = 'expanded.csv'
expand_csv(input_csv, output_csv)
