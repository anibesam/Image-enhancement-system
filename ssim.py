# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
"""
The structural similarity index measure (SSIM) is a method for
predicting the perceived quality of digital television and cinematic pictures,
as well as other kinds of digital images and videos.
SSIM is used for measuring the similarity between two images.
"""
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("AMBE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()
 
 # load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("/home/anibe/Desktop/augustine/sam.jpg")
contrast = cv2.imread("/home/anibe/Desktop/augustine/filtered.png")
shopped = cv2.imread("/home/anibe/Desktop/augustine/equalized.png")
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
# initialize the figure
fig = plt.figure("Images")
images = ("Original Image", original), ("Filtered Image", contrast), ("Equalized Image", shopped)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()
# compare the images
compare_images(original, contrast, "Original vs. Equalized")
compare_images(original, shopped, "Original vs. Equalized")
compare_images(contrast, shopped, "equalized vs. Filtered")