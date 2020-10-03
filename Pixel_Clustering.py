# import imageio
# from imageio import *
from imageio import imread, imsave
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans
import numpy as np

# input variables
k = 4
image_paths = ["Images\CRAYON.jpg","Images\DOG.jpg", "Images\CATAN.jpg"]

#  grab image
for image_path in image_paths:
    # Get image values
    image = imread(image_path)

    # KMeans cluster an image
    image_2d = image.reshape((-1, 3))

    # Big mama KMeans wants to choose how the babies are (eugenics)
    # Big mama makes classifiers
    # tell mama how many buckets the kid will need to carry (# of arms)
    #classifier =  organizer into buckets
    kmeans = KMeans(n_clusters=k)

    # classify to this data
    kmeans.fit(image_2d)

    labels = kmeans.labels_
    # labels = labels.flatten()
    centers =  kmeans.cluster_centers_

    # assign bucket its avg/ center color
    new_image = centers[labels]

    # back to RGB shape
    original_shape = image.shape

    #new_image = new_image.reshape((2048, 2048, 3))
    new_image = new_image.reshape(original_shape)

    # fix data type
    new_image = np.ndarray.astype(new_image, "uint8") 

    # show image
    # plot.imshow(image)
    # plot.show()

    # show image
    # plot.close()
    plot.imshow(new_image)
    plot.show()
