__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud
import time 
import matplotlib.pyplot as plt
from Kmeans import *
from KNN import *


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    # You can start coding your functions here

    def retrieval_by_color(images, labels, colors):
        if isinstance(colors, str):
            colors = [colors]
        coincidences = []
        for image, label in zip(images, labels):
            if all(color in label for color in colors):
                coincidences.append(image)
        return coincidences


    def retrieval_by_shape(images, labels, shape):
        coincidences = []
        for image, label in zip(images, labels):
            if label == shape:
                coincidences.append(images)

        return coincidences


    def retrieval_combined(images, color_labels, shape_labels, colors, shape):
        color_coincidences = retrieval_by_color(images, color_labels, colors)
        shape_coincidences = retrieval_by_shape(images, shape_labels, shape)

        return list(set(color_coincidences) & set(shape_coincidences))


    def Kmeans_statistics(kmeans, images, Kmax):
        """
        Args:
            KMeansClass: Clase Kmeans a la que se le aplicará el análisis.
            images: Conjunto de imágenes para el análisis.
            Kmax: Valor máximo de K que se analizará.

        Return:
            Visualización de las estadísticas obtenidas para cada valor de K.
        """

        WCD_values = []
        convergence_times = []
        iteration_counts = []
        kmeans = KMeans(images, 2)
        k_bona = kmeans.find_bestK(Kmax)

        for k in range(2, Kmax + 1):
            start_time = time.time()
            kmeans.K = k
            kmeans.fit()

            # Calcular el Within-Cluster Distance (WCD)
            WCD = kmeans.withinClassDistance()
            WCD_values.append(WCD)

            # Contar el número de iteraciones
            iterations = kmeans.num_iter
            iteration_counts.append(iterations)

            # Calcular el tiempo de convergencia
            end_time = time.time()
            convergence_time = end_time - start_time
            convergence_times.append(convergence_time)

            print(f"K={k}: WCD={WCD}, Iterations={iterations}, Convergence Time={convergence_time} seconds")

        print("k teorica:", k_bona)
        # Visualizar las estadísticas
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(range(2, Kmax + 1), WCD_values, marker='o')
        plt.title('Within Class Distance (WCD)')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('WCD')

        plt.subplot(1, 3, 2)
        plt.plot(range(2, Kmax + 1), iteration_counts, marker='o')
        plt.title('Number of Iterations')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Iterations')

        plt.subplot(1, 3, 3)
        plt.plot(range(2, Kmax + 1), convergence_times, marker='o')
        plt.title('Convergence Time')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Time (seconds)')

        plt.tight_layout()
        plt.show()

    """
    opcions = {'km_init': 'first'}
    kmeans_instance = KMeans(utils.rgb2gray(train_imgs), 4, opcions)
    Kmeans_statistics(kmeans_instance, utils.rgb2gray(train_imgs), Kmax=10)
    """

    def get_shape_accuracy(predicted_labels, ground_truth_labels):
        """
        Calculate the accuracy of shape prediction.
        :param predicted_labels: Predicted shape labels.
        :param ground_truth_labels: Ground truth shape labels.
        :return: Accuracy as a percentage.
        """
        correct = sum(1 for pred, truth in zip(predicted_labels, ground_truth_labels) if pred == truth)
        total = len(predicted_labels)
        accuracy = (correct / total) * 100
        return accuracy
    

    knn = KNN(utils.rgb2gray(train_imgs), train_class_labels)
    k = 10
    predicted_labels = knn.predict(test_imgs, k)
    shape_accuracy = get_shape_accuracy(predicted_labels, test_class_labels)
    print("Shape Accuracy:", shape_accuracy)

    def Get_color_accuracy(kmeans_results, ground_truth):
            """
            Args:
                kmeans_results: results of the algorithm Kmeans after applying it
                ground_truth: expected colors

            Return:
                Return the accuracy calculated
            """

            total = 0
            for a, b in zip(kmeans_results, list(ground_truth)):
                rao = 1 / len(b)
                if sorted(a) == sorted(b):
                    total += 1
                else:
                    for x in b:
                        if x in a:
                            total += rao
            return 100 * (total / len(kmeans_results))
        
    
    def unique_colors_percentages(colors, percentages):
        """
        Args:
            colors: list of color names obtained from the kmeans object
            percentages: list of percentages

        Return:
            Return the filtered color_list (removed duplicates,
                                          removed irrelevant ...)
                 and the percentages of the colors
        """

        unique_colors = np.unique(colors)

        unique_percentages = []
        for color in utils.colors:
            actual_color_percentages = []
            for c, p in zip(colors, percentages):
                if c == color:
                    actual_color_percentages.append(p)
            if actual_color_percentages:
                actual_color_percentages = np.array(
                    [actual_color_percentages]).mean()
            unique_percentages.append(actual_color_percentages)

        return unique_colors, unique_percentages
    
    


    best_color_labels = []
    best_color_percentages = []

    for image in imgs:
        km = KMeans(image, K=4, options={'km_init': 'first', 'fitting': 'WCD'})
        km.find_bestK(11)


        color_labels = get_colors(km.centroids)
        colorProb = utils.get_color_prob(km.centroids)
        color_percentages = np.max(colorProb, axis=1)

        unique_color_labels, unique_color_percentages = \
            unique_colors_percentages(color_labels, color_percentages)

        best_color_labels.append(unique_color_labels)
        best_color_percentages.append(unique_color_percentages)

    best_color_labels = np.array(best_color_labels, dtype=object)
    best_color_percentages = np.array(best_color_percentages, dtype=object)

    accuracy = Get_color_accuracy(best_color_labels, test_color_labels)





