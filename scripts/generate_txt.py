import os
from PIL import Image

def generate_txt(dataset_dir, output_file):
    index = 1
    # Walk through the dataset directory
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.jpg'):
                    # Get the relative path of the jpg file without extension
                    relative_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    folder_name, filename = os.path.split(relative_path)
                    base_filename = os.path.splitext(filename)[0]  # Remove .jpg extension

                    # Construct the required image_id format: folder/filename
                    image_path = os.path.join(folder_name, base_filename)

                    absolute_jpg_path = os.path.join(root, file)

                    # Verify the image integrity
                    try:
                        with Image.open(absolute_jpg_path) as img:
                            img.convert('RGB')  # Ensure the image is not corrupted
                    except (OSError, IOError):
                        print(f"Skipping corrupted image: {absolute_jpg_path}")
                        continue

                    # Get the corresponding txt file path
                    txt_file = base_filename + '.json'
                    txt_path = os.path.join(root, txt_file)

                    # Check if the corresponding txt file exists
                    if not os.path.exists(txt_path):
                        print(f"Skipping corrupted image: {absolute_jpg_path}")
                        continue

                    # Write the relative_path to the txt file
                    f.write(image_path + '\n')
                    f.flush()

                    index = index + 1

                    if index % 100000 == 0:
                        print(index)

if __name__ == "__main__":
    # Example usage
    data_directory = "/scratch/xianfeng/text-to-image-2M/data_512_2M/untar"  # Replace with your dataset directory
    output_txt_file = "dataset.txt"  # Replace with your desired output TXT file name
    generate_txt(data_directory, output_txt_file)
    print(f"TXT file '{output_txt_file}' generated successfully!")