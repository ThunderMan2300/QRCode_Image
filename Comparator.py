import cv2
import os
import subprocess
import numpy as np

def convert_to_jbig2_image(file_name, target_folder, image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image to bitonal using Otsu's thresholding
    _, bitonal_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the bitonal image temporarily for the jbig2enc processing
    temp_path = os.path.join(target_folder, "temp.png")
    cv2.imwrite(temp_path, bitonal_image)

    base_name, _ = os.path.splitext(file_name)
    jbig2_path = os.path.join(target_folder, base_name + ".jbig2")
    sym_file = os.path.join(target_folder, base_name + "_sym")

    # Use jbig2enc to encode the image
    subprocess.run(['jbig2enc', '-s', '-o', sym_file, temp_path])

    # Combine the symbol table and the page data into a single compressed .jbig2 file
    subprocess.run(['cat', sym_file + '0001', sym_file + '.sym', '>', jbig2_path])

    # Clean up temporary files
    os.remove(temp_path)
    os.remove(sym_file + '0001')
    os.remove(sym_file + '.sym')

def rle_encode(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten the image and compute the RLE
    pixels = gray_image.flatten()
    runs = []
    run_val = pixels[0]
    run_len = 1

    for i in range(1, len(pixels)):
        if pixels[i] == run_val:
            run_len += 1
        else:
            runs.append((run_val, run_len))
            run_val = pixels[i]
            run_len = 1
    runs.append((run_val, run_len))

    return runs

def convert_to_rle_image(file_name, target_folder, image):
    # Encode the image using RLE
    encoded_data = rle_encode(image)

    # Save the encoded data
    base_name, _ = os.path.splitext(file_name)
    rle_path = os.path.join(target_folder, base_name + "_rle.txt")

    with open(rle_path, 'w') as f:
        for val, count in encoded_data:
            f.write(f"{val} {count}\n")

    # This function saves the RLE encoded data as a text file.
    # Each line of the file contains a value and its run length, separated by a space.

def encode_dpcm(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Encode using DPCM
    encoded = np.zeros_like(gray_image)
    encoded[0, 0] = gray_image[0, 0]
    for i in range(1, gray_image.shape[0]):
        for j in range(1, gray_image.shape[1]):
            predicted_value = gray_image[i-1, j-1]
            encoded[i, j] = gray_image[i, j] - predicted_value
    
    return encoded

def compress_with_dpcm(file_name, target_folder, image):
    # Encode the image using DPCM
    encoded_image = encode_dpcm(image)
    
    # Save the encoded image
    base_name, _ = os.path.splitext(file_name)
    encoded_path = os.path.join(target_folder, base_name + "_dpcm.png")
    cv2.imwrite(encoded_path, encoded_image)

def covert_to_jpeg_image(file_name, target_folder, image):
    base_name, _ = os.path.splitext(file_name)
    jp2_path = os.path.join(target_folder, base_name + ".jp2")
    cv2.imwrite(jp2_path, image)

def convert_to_lossless_webp_image(file_name, target_folder, image):
    # Save the image temporarily in the target folder
    temp_path = os.path.join(target_folder, file_name)
    cv2.imwrite(temp_path, image)

    # Convert the image to lossless WebP
    base_name, _ = os.path.splitext(file_name)
    webp_path = os.path.join(target_folder, base_name + ".webp")
    
    # Use the cwebp tool to convert the image in lossless mode
    subprocess.run(['cwebp', '-lossless', temp_path, '-o', webp_path])
    
    # Remove the temporary image
    os.remove(temp_path)

def convert_to_jpegxl_image(file_name, target_folder, image):
    # Save the image temporarily in the target folder
    temp_path = os.path.join(target_folder, file_name)
    cv2.imwrite(temp_path, image)

    # Convert the image to JPEG XL
    base_name, _ = os.path.splitext(file_name)
    jxl_path = os.path.join(target_folder, base_name + ".jxl")
    
    # Use the JPEG XL tool to convert the image (assuming 'cjxl' is the command-line tool for encoding)
    subprocess.run(['cjxl', temp_path, jxl_path])
    
    # Remove the temporary image
    os.remove(temp_path)

def convert_to_bpg_image(file_name, target_folder, image):
    # Save the image temporarily in the target folder
    temp_path = os.path.join(target_folder, file_name)
    cv2.imwrite(temp_path, image)

    # Convert the image to BPG
    base_name, _ = os.path.splitext(file_name)
    bpg_path = os.path.join(target_folder, base_name + ".bpg")
    
    # Use the BPG encoder to convert the image
    subprocess.run(['bpgenc', temp_path, '-o', bpg_path])
    
    # Remove the temporary image
    os.remove(temp_path)

def convert_to_jpegls_image(file_name, target_folder, image):
    # Save the image temporarily in the target folder
    temp_path = os.path.join(target_folder, file_name)
    cv2.imwrite(temp_path, image)

    # Convert the image to JPEG LS
    base_name, _ = os.path.splitext(file_name)
    jpls_path = os.path.join(target_folder, base_name + ".jls")
    
    # Use the JPEG LS tool to convert the image (assuming 'charls' is the command-line tool)
    subprocess.run(['charls', '-i', temp_path, '-o', jpls_path])
    
    # Remove the temporary image
    os.remove(temp_path)

def convert_and_save_images(file_name, target_folder, image):
    covert_to_jpeg_image(file_name, target_folder, image)
    #convert_to_jpegls_image()          Not supported on Windows    charls
    #convert_to_bpg_image()             Not supported on Windows    bpgenc
    #convert_to_jpegxl_image()          Not supported on Windows    cjxl
    #convert_to_lossless_webp_image()   Not supported on Windows    cwebp
    #convert_to_jbig2_image()           Not supported on Windows    jbig2enc
    compress_with_dpcm(file_name, target_folder, image) # Only for grayscale images
    convert_to_rle_image(file_name, target_folder, image)  # Only for grayscale images

def load_and_save_image(source_folder, target_folder, file_name):
    # Load the image from the source folder
    image_path = os.path.join(source_folder, file_name)
    image = cv2.imread(image_path)
    
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Save the image to the target folder
    target_path = os.path.join(target_folder, file_name)
    cv2.imwrite(target_path, image)

    convert_and_save_images(file_name, target_folder, image)
   
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
