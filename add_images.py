import os
from PIL import Image

#Source directories
source_imdir = "ChinaSet_AllFiles/CXR_png"
source_labdir = "ChinaSet_AllFiles/ClinicalReadings"

#The directory in which we will save the new images
target_dir = "chest_xray/train/NORMAL" 

#Source images and clinical readings (labels)
source_images = os.listdir(source_imdir)
source_labels = os.listdir(source_labdir)

#Extract the name of the image, to match with the appropriate clinical reading
def extract_name(string):
    name = string.split(".")[0]
    return name

#Find the corresponding image reading
def find_reading(image_name, readings):
    for reading in readings:
        if image_name in reading:
            return reading
        


for image in source_images:
    #Extract the name of the image
    image_name = extract_name(image)
    #find the coresponding .txt reading
    image_reading = find_reading(image_name, source_labels)
    #Open the .txt reading and check if its normal
    reading_path = os.path.join(source_labdir, image_reading)
    with open(reading_path) as reading:
        #If the reading is normal, we save the corresponding image
        if "normal" in reading.read():
            image_path = os.path.join(source_imdir, image)
            im = Image.open(image_path)
            #Make a copy of the image
            new_im = im.copy()
            #Save it in the new directory
            new_im.save(os.path.join(target_dir, image))
            reading.close()
        else:
            continue



        