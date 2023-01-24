from numpy import random, sqrt, min, max, argmin, sum, array, mean, isnan, not_equal
import matplotlib.pyplot as plt
from time import sleep
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from time import time

class Kmeans:

    def __init__(self, cluster_amount=3, max_iteration=100):
        self.ion = False
        self.centroids = []
        self.cluster_amount = cluster_amount
        self.max_iteration = max_iteration
        self.boundary_min = 0
        self.boundary_max = 0
        self.ax = None
        self.polygon = None
        self.points = []
        with plt.ion():
            self.kmeans_figure, self.axes0 = plt.subplots(1, 2, figsize = (12,6))
            self.kmeans_figure.tight_layout(pad=4.0)
            self.kmeans_figure.suptitle(f'Iterations={max_iteration}', fontsize=16)

    def calculateClustersAmount(self, avg_dist_in_cluster, avg_dist_to_nearest_cluser):
        silhouette_coeff = (avg_dist_to_nearest_cluser - avg_dist_in_cluster)/max(avg_dist_in_cluster,avg_dist_to_nearest_cluser)
        return silhouette_coeff # fun fact

    def distribiutingCentroids(self, points, n, param):
        '''
        Distribiuting centroids according to chosen param value, to observe to most successful solution
        '''
        if param=="0,0":
            centroids = [array([0,0]) for i in range(n)]
            print(centroids)
            return centroids
        elif param=="re-evaluate":
            print(len(points))
            i = random.randint(0, len(points))
            centroids = [points[i]]
            for _ in range(self.cluster_amount-1):
                dists = sum([self.calculateEuclideanDist(centroid, points) for centroid in self.centroids], axis=0) # calculates every distance between point and centroid
                dists /= sum(dists) # normalizes distances
                new_centroid_idx, = random.choice(range(len(dists)), size=1, p=dists) # basing on distances we choose the most probable point
                centroids.append(points[new_centroid_idx])
            return centroids
        else:
            self.boundary_min,self.boundary_min = min(points, axis=0), max(points,axis=0)
            centroids = [random.uniform(self.boundary_min,self.boundary_max) for i in range(n)]
            return centroids 

    def calculateEuclideanDist(self, point, centroids):
        return sqrt(sum((array(point)-array(centroids))**2, axis=1)) #axis=1 allows to compute a distance for each centroid in refrence to chosen point

    def displayFigure(self, centroids, sorted_points):
        if self.ion:
            self.kmeans_figure.clf()
            plt.waitforbuttonpress()
        for group in sorted_points:
            self.axes0[0].plot([x for x,y in group], [y for x,y in group], 'o', markersize=5)
        self.axes0[0].plot([x for x,y in centroids], [y for x,y in centroids], '+', markersize=10)
        self.axes0[0].set_title("KMeans")
        print(centroids)
        plt.draw()
        plt.pause(0.01)
        
    
    def groupPoints(self, points, scatter=False):
        self.centroids = self.distribiutingCentroids(points, self.cluster_amount, "re-evaluate")
        prev_centroids = None
        self.points = points
        
        iteration = 0
        if scatter:
            self.ion = True
            self.axes0[0].scatter([x for x,y in points], [y for x,y in points], facecolors='none', edgecolors='k')  
        while not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iteration:
            sorted_points = [[] for i in range(self.cluster_amount)]
            for point in points:
                dist = self.calculateEuclideanDist(point, self.centroids) 
                group = argmin(dist)
                sorted_points[group].append(point) #We group up gathered points to nearest centroids
            if iteration > 0 and scatter:                    #Uncomment to see visualization step by step
                self.displayFigure(self.centroids, sorted_points)
            prev_centroids = self.centroids
            self.centroids = [mean(cluster, axis=0) for cluster in sorted_points] #We recalculate new clusters, accordingly
                #to average euclidean distance of grouped points 
            for i, centroid in enumerate(self.centroids): #We assign numbers to corresponding clusters
                if isnan(centroid).any():  #If any centroid remains empty we replace it in prev_centroids 
                    #to keep loop conditions consistent
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        self.displayFigure(self.centroids, sorted_points)
        return sorted_points

centers = 5
max_iteration = 100
points = []
for n in range(500):
    x = float(random.rand(1,1)*100)
    y = float(random.rand(1,1)*100)
    points.append((x,y))

points0, labels = make_blobs(n_samples=100, centers=5, random_state=36)
points0 = StandardScaler().fit_transform(points0)
X0 = [X[0] for X in points0]
Y0 = [X[1] for X in points0]

pointsPaired = list(zip(X0, Y0))

#for presentation purpose random_state=42

start = time()
kmeans = Kmeans(centers,max_iteration)
kmeans.groupPoints(pointsPaired,True)
passed0 = time()-start

start = time()
test0 = KMeans(n_clusters=centers, max_iter=max_iteration).fit(points0)
passed1 = time()-start

kmeans.axes0[1].scatter(x = X0, y = Y0, c = test0.labels_, s = 25)
kmeans.axes0[1].scatter(test0.cluster_centers_[:, 0], test0.cluster_centers_[:, 1], s=100, marker='+', c='gray', label='centroids')
kmeans.axes0[1].set_title("KMeans (sklearn)")

plt.show()
