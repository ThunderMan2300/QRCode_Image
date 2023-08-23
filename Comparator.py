import cv2
import os

def covert_to_jpeg_image(file_name, target_folder, image):
    base_name, _ = os.path.splitext(file_name)
    jp2_path = os.path.join(target_folder, base_name + ".jp2")
    cv2.imwrite(jp2_path, image)

def load_and_save_image(source_folder, target_folder, file_name):
    # Load the image from the source folder
    image_path = os.path.join(source_folder, file_name)
    image = cv2.imread(image_path)
    
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    covert_to_jpeg_image(file_name, target_folder, image)
    
    # Save the image to the target folder
    target_path = os.path.join(target_folder, file_name)
    cv2.imwrite(target_path, image)

def display_image_sizes(target_folder):
    # Display the header
    header = "Image Name".ljust(20) + "Extension".ljust(15) + "Size (bytes)"
    print(header)
    print('-' * len(header))
    
    # List all files in the target folder
    image_names = os.listdir(target_folder)
    
    # Display the details for each image
    for image_name in image_names:
        base_name, ext = os.path.splitext(image_name)
        file_size = os.path.getsize(os.path.join(target_folder, image_name))
        
        print(base_name.ljust(20) + ext.ljust(15) + str(file_size))


if __name__ == "__main__":
    # Define source and target folders
    source_folder = "assets"
    target_folder = "results"
    
    # List of image names
    image_names = ["chunked_image.png", "id_sample.png", "noise_image.png"]

    
    # Load and save each image
    for image_name in image_names:
        load_and_save_image(source_folder, target_folder, image_name)

    # Display the sizes in a table format
    display_image_sizes(target_folder)
