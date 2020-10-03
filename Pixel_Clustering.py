# import imageio
# from imageio import *
from imageio import imread, imsave
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans

#  grab image
image_path = "Images\DOG.jpg"
image = imread(image_path)

# KMeans cluster an image
k = 3
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

new_image = centers[labels]

# print(image_2d.shape)

# show image
# plot.imshow(image)
# plot.show()

# show image
# plot.close()
# plot.imshow(new_image)
# plot.show()
