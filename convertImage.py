from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

root_dir = '../Simple-Sign-Language-Detector/mydata/training_set'

train_data = [[] for x in range(2)]

folderNames = []
for letterEntry in os.scandir(root_dir):
    if not letterEntry.name.startswith('.') and letterEntry.is_dir():

        # Add label array to training_data[1]
        label_arr = np.zeros(26)
        label_arr[ord(letterEntry.name)-65] = 1
        train_data[1].append(label_arr)

        for img in os.scandir(letterEntry.path):
            lc_img = Image.open(img.path).convert('L')
            img_arr = np.asarray(lc_img)
            train_data[0].append(img_arr.flatten())

            

for i in range(3):
    plt.figure()
    plt.imshow(np.reshape(train_data[0][i], (64, 64)),
               cmap="gray")
    plt.axis('off')
    plt.title(str(np.argmax(train_data[1][i])))
    plt.show()
# img = Image.open('./A_test.png').convert('L')
# image_as_np = np.asarray(img)
# print(image_as_np.size)
