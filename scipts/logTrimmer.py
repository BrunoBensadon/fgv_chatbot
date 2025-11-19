import os

def filter_log_file(log_file_path):
    """
    Reads a log file, removes lines containing "change detected" 
    or "changes detected", and overwrites the original file.
    """
    lines_to_keep = []
    try:
        # Step 1: Read all lines from the file
        with open(log_file_path, 'r') as infile:
            for line in infile:
                # Step 2: Check if the line contains the target strings
                if "change detected" not in line.lower() and "changes detected" not in line.lower():
                    lines_to_keep.append(line)
        
        # Step 3: Write the filtered lines back to the same file
        with open(log_file_path, 'w') as outfile:
            outfile.writelines(lines_to_keep)
        
        print(f"Lines containing 'change detected' or 'changes detected' have been removed from {log_file_path}.")

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the path to your log file here
log_file_name = './messages.log' 
filter_log_file(log_file_name)
