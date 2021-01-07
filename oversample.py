import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplim
import os
import random 
import PIL
from PIL import ImageEnhance
import cv2
import pathlib

## Investigate classes in training and test sets
num_normal_train = len(os.listdir(f"{os.getcwd()}/chest_xray/train/NORMAL/"))
num_pneumonia_train = len(os.listdir(f"{os.getcwd()}/chest_xray/train/PNEUMONIA/"))
num_normal_test = len(os.listdir(f"{os.getcwd()}/chest_xray/test/NORMAL/"))
num_pneumonia_test = len(os.listdir(f"{os.getcwd()}/chest_xray/test/PNEUMONIA/"))

print("{} normal images in train set".format(num_normal_train))
print("{} pneumonia images in train set".format(num_pneumonia_train))
print("{} normal images in test set".format(num_normal_test))
print("{} pneumonia images in test set".format(num_pneumonia_test))

#Test the effect of Blurring
#Open existing image
OriImage = PIL.Image.open("chest_xray/train/NORMAL/NORMAL2-IM-1356-0001.jpeg")
plt.subplot(121),plt.imshow(OriImage, cmap="gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])

blurImage = OriImage.filter(PIL.ImageFilter.BoxBlur(8))
plt.subplot(122),plt.imshow(blurImage, cmap="gray"),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#Test the effect of brighness adjustment
plt.subplot(121),plt.imshow(OriImage, cmap="gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])

#Adjusting image brightness
#Values larger than 1 add brightness to the image
factor = 1.2
brightness = PIL.ImageEnhance.Brightness(OriImage)
enhanced_im = brightness.enhance(factor)

plt.subplot(122),plt.imshow(enhanced_im, cmap="gray"),plt.title('Enhanced')
plt.xticks([]), plt.yticks([])
plt.show()

#We are going to apply random oversampling for the under-represented class "NORMAL"
#We will copy random images of the class until we have a roughly balanced result.
#For every random image we choose, we apply either blurring or brightness enhancement.
#The transformation is chosen randomly as well.
#We will generate 2000 copies of random normal xrays
file_list = os.listdir(f"{os.getcwd()}/chest_xray/train/NORMAL/")

for repetition in range(0,2000):
  #select a random image from class NORMAL in the training set
  random_file = random.choice(file_list)
  #get image extension
  extension = pathlib.Path(f"{os.getcwd()}/chest_xray/train/NORMAL/" + random_file).suffix
  #Read the image
  im = PIL.Image.open(f"{os.getcwd()}/chest_xray/train/NORMAL/" + random_file)
  #List of possible transformations
  transformation = random.choice(["blur", "enhance"])
  #Check images mode
  if im.mode != "L":
    #converting pixel depth to 1
    im = im.convert("L")
  #Apply transformation
  if transformation == "blur":
    new_im = im.filter(PIL.ImageFilter.BoxBlur(8)) 
  else:
    brightness = PIL.ImageEnhance.Brightness(im)
    new_im = brightness.enhance(1.2) 
  #Save the new image
  new_im.save(f"{os.getcwd()}/chest_xray/train/NORMAL/" + str(repetition) + extension)

num_normal_train_new = len(os.listdir(f"{os.getcwd()}/chest_xray/train/NORMAL/"))
print("{} normal images in train set".format(num_normal_train_new))
print("{} pneumonia images in train set".format(num_pneumonia_train))