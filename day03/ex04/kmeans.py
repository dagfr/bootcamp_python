"""
    Exercise from 42AI : https://github.com/42-AI/bootcamp_python/blob/master/day03/ex04/ex04.md
    Resource : https://raw.githubusercontent.com/42-AI/bootcamp_python/master/day03/resources/solar_system_census.csv
    KMean Calculation
"""

import numpy as np
import os
import math
import random


def avg(li):
    """avg 
    Returns the average value of a list

    Args:
        li ([list]): list of float value

    Returns:
        [float]: average value of the list
    """
    return sum(li) / len(li) 

def read_data(d):
    """read_data 
    Read a CSV file with 4 columns : int,float,float,float and returns a dict of the values 
    with key the int of the csv and an Numpy Array of the 3 floats as value
    
    

    Args:
        d ([path like str]): the path to the file

    Returns:
        [dict]: value from the csv file as a dict. Len = number of line -1 (header) Key = First column(int)
        Value : Numpy Array of the 3 floats
        ie. {
            '1' => Array([190.2, 85.3, 0.95]),
            '2' => Array([178.2, 81.8, 0.87])
        }
        Returns None if file doesn't exists

    """
    if not os.path.exists(d):
        return None
    points = {}
    with open(d,"r",encoding='utf-8') as f:
        line = f.readline() # skip header
        line = f.readline()
        while line:
            data = line.strip().split(',') #remove \n and split value using ','
            points[data[0]] = np.array(data[1:]) #first item is key other is converted to Numpy array as the value
            line = f.readline() #next line
    return points

def pick(d,n):
    """pick
    Returns n different random keys from dict d.
    This method is used to get random centroids from the existing points
    TO DO : use choices out of a loop ?

    Args:
        d ([dict]): Dict of value as returned by read_data
        n ([int]): Number of random value needed

    Returns:
        [list]: list of size n with random keys from the dict d
    """
    r = []
    #r = random.choices(list(d),n) #I didn't manage to use choices
    while len(r) < n:
        a = random.choice(list(d))
        if a not in r:
            r.append(a)
    return r

def mean_point(points):
    """mean_point 
    Returns the gravity center of a list of points.
    Gravity center is the point with mean coordinate of the list

    Args:
        points ([list]): a list of Point

    Returns:
        [Point]: the gravity center of all Points of the list points.
    """
    return Point([avg([p.x for p in points]),avg([p.y for p in points]),avg([p.z for p in points])])

class Point:
    """ Point class
    A class to represent a being with height,weight,bone_density as a coordinate in a 3-Dimensional plan
    (height,weight,bone_density) is the coordinate in the plan
    """
    def __init__(self,coord):
        """__init__ 
        Create the point from coord

        Args:
            coord ([list]): A list of str standing for height,weight,bone_density of a being from the solar system
            data comes from a csv file and is string and needs to be casted to float
        """
        self.x, self.y, self.z=float(coord[0]),float(coord[1]),float(coord[2])

    def distance(self,p):
        """distance 
        Returns the distance between Point self and given p Point. 
        Distance between 2 points (x1,y1,z1) and (x2,y2,z2) in a 3 Dimensional plan is :
             sqrt( (x1-x2)² + (y1-y2)² + (y1-y2)² ) 
        N.B. ² will ensure valeur are positive float and don't need to be 'abs' or 'max - min'

        This whole class could have been replaced by a distance(a,b) function with 2 Numpy Arrays

        Args:
            p ([Point]): the Point to calculate the distance self is from

        Returns:
            [float]: the distance in the same unit point are
        """
        return math.sqrt((self.x-p.x)**2 + (self.y-p.y)**2 + (self.z-p.z)**2)

    def __str__(self):
        """__str__ Overriding magic ___str___ function to represent a point as a tuple of 3 coordinates

        Returns:
            [str]: str representation of Point is (x,y,z)
        """
        return "({},{},{})".format(self.x, self.y, self.z)

class KmeansClustering:
    """ KmeansClustering class
    This class represent a example of kmean calculation for a list of points
    For more information on Kmean : https://bigdata-madesimple.com/possibly-the-simplest-way-to-explain-k-means-algorithm/

    """
    def __init__(self, max_iter=20, ncentroid=5):
        """__init__ 
        Creates an instance of the class KmeansClustering

        Args:
            max_iter (int, optional): Maximum number of calculation the user want to do. Defaults to 20.
            ncentroid (int, optional): Number of centroid to gather data. Defaults to 5.
        """
        self.ncentroid = ncentroid # number of centroids
        self.max_iter = max_iter # number of max iterations to update the centroids
        self.centroids = [] # list of each centroid (Point)
        self.clusters = [[] for _ in range(self.ncentroid)] #list of each being (Point) in the clusters. 
        self.names = ["beings" for _ in range(self.ncentroid)] #list of each name of clusters. 
        """
            centroids, clusters and names are sharing indexes, the cluster at index 1 
            is the list of Points related to the centroid at index 1
            this cluster's name is the name at index 1
        """
    
    def __str__(self):
        """__str__ 
        Returns the string representation of a data set for debugging/validating purpose

        Returns:
            [str]: str representation of KmeansClustering
        """
        return "Will iterate {} times through {} centroids :\n   {}".format(
            self.max_iter, 
            self.ncentroid,
            # print centroid coordinates, size and name of the related cluster for each centroid
            "\n   ".join([ self.centroids[i].__str__() + " => " + str(len(self.clusters[i]))+ " "+self.names[i] for i in range(self.ncentroid)]))
    
    def setCentroids(self,centroids):
        """setCentroids 
        Sets new centroids randomly at first and calculated (with the mean_point function) after beings' assignment

        Args:
            centroids ([list]): list of Points to represent centroids
        """
        self.centroids = centroids

    def fit(self, X):
        """fit 
        Assign beings (Point) to each cluster and then move centroid at the gravity center of the cluster
        This process is repeated max_item times or stop if beings are evenly distributed(+/-20%)
        Then it calculates names with the provided Hints : 
            people from the Belt are taller and with the lowest bone density
            then taller are from Mars
            then slender are from Venus
            rest is from Earth 


        Args:
            X ([type]): [description]
        """
        print("Kmean calculation ...")
        mi = int((len(X)/self.ncentroid)/1.2) #min value to be +/-20% from the median
        ma = int((len(X)/self.ncentroid)*1.2) #max value to be +/-20% from the median
        self._fit_once(X) #perform the assignement once

        for _ in range(1,self.max_iter):
            a = np.array([len(b) for b in self.clusters])
            b =(a>mi) & (a<ma)
            if all(b):
                #save centroids and breaks if evenly distributed
                self._save_centroids() 
                break
            #else move centroids to center of the clusters and repeat the assignement
            self._move_centroids()
            self._fit_once(X)
        
        #then calculate names
        self._calc_names()

    def _fit_once(self, X):
        """_fit_once 
        Perform one assignement for beings from list X to clusters
        each Point is assigned to the cluster with the clostest centroids (uses the distance Point method)

        Args:
            X ([list]): list of Point
        """
        for i in [Point(p) for p in X.values()]:
            min_idx = 999
            min_distance = 9999999
            for j in range(len(self.centroids)):
                #loop through centroid to find the closest and store the index of it in min_idx
                c = self.centroids[j]
                if i.distance(c)<min_distance:
                    min_distance = i.distance(c)
                    min_idx = j
            #append the Point to the cluster
            self.clusters[min_idx].append(i)

    def _move_centroids(self):
        """_move_centroids 
            Changes centroids to gravity center of clusters
            Empties clusters to allow recalculation
        """
        self._save_centroids()
        self.clusters = [[] for _ in range(self.ncentroid)]

    def _save_centroids(self):
        """_save_centroids 
            Changes centroids to gravity center of clusters
            it "map" mean_point to each cluster to get the mean point
        """
        self.setCentroids([mean_point(self.clusters[i]) for i in range(self.ncentroid)])

    def _calc_names(self):
        """_calc_names 
        Calculates the name according to hints and store them in the names attribute
        """
        idx = 99
        val = 5
        
        for i in range(self.ncentroid):
            if self.names[i] == 'beings':
                cl = self.clusters[i]
                if avg([j.z for j in cl])<val:
                    idx =i
                    val = avg([j.z for j in cl])
        #First get the lowest bones density for the belt
        self.names[idx]="Citizens of the Belt"
        idx = 99
        val = 0
        for i in range(self.ncentroid):
            if self.names[i] == 'beings':
                cl = self.clusters[i]
                if avg([j.x for j in cl])>val:
                    idx =i
                    val = avg([j.x for j in cl])
        #Then the tallest for Mars
        self.names[idx]="People of the Martian Republic"
        idx = 99
        val = 0
        for i in range(self.ncentroid):
            if self.names[i] == 'beings':
                cl = self.clusters[i]
                if avg([j.y for j in cl])>val:
                    idx =i
                    val = avg([j.y for j in cl])
        #Finally slendest are from Venus and other from Earth
        self.names[idx]="Terranians"
        self.names = ["People from Venus" if i == "beings" else i for i in self.names]


if __name__ == "__main__":
    """
    Tests
    get the file
    pick 4 random points
    create a dataset
    set random points as initial centroids
    perform the fit method
    print the result
    """
    a = read_data(os.path.join(os.path.dirname(__file__),'data.csv'))
    r = pick(a,4)
    k = KmeansClustering(ncentroid=4)
    k.setCentroids([Point(a[i]) for i in r])
    print(k)
    k.fit(a)
    print(k)
