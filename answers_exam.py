from utils_data import read_dataset, visualize_k_means
from Kmeans import *
from KNN import KNN

IDX_IMAGE = 10

if __name__ == '__main__':

    # import images with color
    train_imgs, _, train_color_labels, test_imgs, _, test_color_labels = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json', with_color=True
    )

    # import images in grayscale
    train_imgs_grayscale, train_class_labels, _, test_imgs_grayscale, test_class_labels, _ = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json', with_color=False
    )

    # BELOW HERE YOU CAN CALL ANY FUNCTION THAT YOU HAVE PROGRAMED TO ANSWER THE QUESTIONS OF THE EXAM #
    # this code is just for you, you won't have to upload it after the exam #

    # this is an example of how to call some functions that you have programed
    km = KMeans(test_imgs[IDX_IMAGE], K=4)
    km.fit()
    visualize_k_means(km, (80, 60, 3))
"""    
km=KMeans(test_imgs[0],K=3)
km.fit()
print(km.centroids, km.withinClassDistance())   
ans:d     
"""

"""
knn = KNN(train_imgs_grayscale, train_class_labels)
knn.get_k_neighbours(np.array([test_imgs_grayscale[0]]), 3)
print(knn.neighbors)
"""

#ans: c     
#4
#test_img = test_imgs[100]
#point_RGB = test_img[52,32].reshape(-1,3)
#print(get_colors(point_RGB))

"""
#5
km = KMeans(test_imgs[IDX_IMAGE], K=3)
km.fit()
print(km.centroids)
"""
    
#6
km = KMeans(test_imgs[40], K=3)
print(km.find_bestK(10))
km = KMeans(test_imgs[40], 4)
km.fit()
print(get_colors(km.centroids))

'''
#8
knn = KNN(train_imgs_grayscale, train_class_labels)

sol = knn.predict(test_imgs_grayscale[:50],5)
print(sum([i=="Flip Flops" for i in sol]) / sol.shape[0])
knn = KNN(train_imgs_grayscale, train_class_labels)

sol = knn.predict(np.array([test_imgs_grayscale[0]]),3)
print(sol)
'''

