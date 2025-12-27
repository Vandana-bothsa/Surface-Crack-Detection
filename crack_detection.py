import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from zipfile import ZipFile
uploaded_path = "/content/archive.zip"

with ZipFile(uploaded_path, 'r') as zip_ref:
    zip_ref.extractall("/content/crack_dataset")

IMG_SIZE = 64
def load_images(folder, limit=200):  # Limit number to avoid RAM crash
    images = []
    count = 0
    for file in sorted(os.listdir(folder)):
        if count >= limit: break
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        count += 1
    return np.array(images)

pos_dir = "/content/crack_dataset/Positive"
neg_dir = "/content/crack_dataset/Negative"

X = load_images(pos_dir)
y = load_images(neg_dir)

X = X / 255.0
y = y / 255.0
y = np.expand_dims(y, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def mini_unet(input_size=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_size) # Explicitly define shape

    c1 = layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(16, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D()(c3)
    m1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(m1)

    u2 = layers.UpSampling2D()(c4)
    m2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(8, 3, activation='relu', padding='same')(m2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

model = mini_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from skimage.morphology import skeletonize
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Predict
preds = model.predict(np.expand_dims(X_test, axis=-1))

# Select one image from test set
i = 0
pred_mask = (preds[i].reshape(IMG_SIZE, IMG_SIZE) > 0.5).astype(np.uint8)

# Skeletonize to thin crack line
skeleton = skeletonize(pred_mask).astype(np.uint8)

# Find contours from the skeleton
contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert original grayscale image to BGR
input_image = (X_test[i] * 255).astype(np.uint8)
crack_highlighted = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

# Draw thin red contours on the image
cv2.drawContours(crack_highlighted, contours, -1, (0, 0, 255), thickness=1)  # red color

# Show result
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(crack_highlighted, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Crack Highlighted in Thin Red Contour")
plt.show()

# Step 1: Load your uploaded image
image_path = "/content/crack_img.png"  # replace with your uploaded file path
img = cv2.imread(image_path)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Use Canny edge detection to detect crack lines
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Step 5: Find contours from the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw red contours (cracks) on the original image
crack_highlighted = img.copy()
cv2.drawContours(crack_highlighted, contours, -1, (0, 0, 255), thickness=1)  # red color

# Step 7: Show output
plt.figure(figsize=(6,6))
plt.title("Crack Marked with Red Line")
plt.imshow(cv2.cvtColor(crack_highlighted, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Step 1: Load your uploaded image
image_path = "/content/crack.png"  # replace with your uploaded file path
img = cv2.imread(image_path)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Use Canny edge detection to detect crack lines
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Step 5: Find contours from the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw red contours (cracks) on the original image
crack_highlighted = img.copy()
cv2.drawContours(crack_highlighted, contours, -1, (0, 0, 255), thickness=1)  # red color

# Step 7: Show output
plt.figure(figsize=(6,6))
plt.title("Crack Marked with Red Line")
plt.imshow(cv2.cvtColor(crack_highlighted, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
