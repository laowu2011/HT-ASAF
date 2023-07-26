import os

# Define a function to truncate the content between 'module TSC' and 'endmodule'
def truncate_module(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        start_index = content.find('module TSC')
        end_index = content.find('endmodule', start_index)
        truncated_content = content[start_index:end_index + 9]  # +9 to include 'endmodule' in the truncated content
    return truncated_content

# Create the 'Augment_sample' folder if it doesn't exist
if not os.path.exists("Augment_sample"):
    os.mkdir("Augment_sample")

# Process files in 'Augment_sample_undeal' folder
file_list = os.listdir("Augment_sample_undeal")
for filename in file_list:
    file_path = os.path.join("Augment_sample_undeal", filename)
    if os.path.isfile(file_path):
        # Truncate the module content and save it in the 'Augment_sample' folder
        truncated_content = truncate_module(file_path)
        with open(os.path.join("Augment_sample", filename), "w") as f:
            f.write(truncated_content)

print("Truncation and saving completed.")