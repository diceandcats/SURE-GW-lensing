"""SURE cluster lensing module."""

from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy.optimize._minimize as minimize


class ClusterLensing:
    """
    Class to get the lensing properties of a cluster by deflection and lens potential map of the cluster.
    """

    def __init__(self, alpha_map_x, alpha_map_y, lens_potential_map, z_l , z_s, pixscale, size, x_src, y_src):
        """
        Parameters:
        ---------------
        deflection_map_x: The deflection map in x direction in arcsec.
        deflection_map_y: The deflection map in y direction in arcsec
        lens_potential_map: The lens potential map in arcsec^2.
        x_src: The x coordinate of the source in arcsec.
        y_src: The y coordinate of the source in arcsec.
        z_l: The redshift of the lens.
        z_s: The redshift of the source.
        """
        self.alpha_map_x = alpha_map_x / pixscale    #in pixel now
        self.alpha_map_y = alpha_map_y / pixscale    #in pixel now
        self.lens_potential_map = lens_potential_map
        self.z_l = z_l
        self.z_s = z_s
        self.pixelscale = pixscale
        self.size = size
        self.x_src = x_src
        self.y_src = y_src
        self.image_positions = None
        self.magnifications = None
        self.time_delays = None


    def find_rough_def_pix(self):    # data are in pixel
        """
        Find the pixels that can ray-trace back to the source position roughly.
        """
        alpha_x = self.alpha_map_x   # make sure alpha_x and alpha_y are in pixel
        alpha_y = self.alpha_map_y
        coord = (self.x_src, self.y_src)
        coord_x_r, coord_y_r = coord[0] % 1, coord[1] % 1
        y_round, x_round = round(coord[1]), round(coord[0])

        # Pre-calculate possible matching rounded values for efficiency
        y_possible_rounds = {y_round, y_round - 1} if coord_y_r == 0.5 else {y_round}
        x_possible_rounds = {x_round, x_round - 1} if coord_x_r == 0.5 else {x_round}

        coordinates = []
        n = 0
        size = self.size
        # Iterate over a pre-defined range, assuming alpha_y_2d and alpha_x_2d are indexed appropriately
        for i in range(size):
            for j in range(size):
                ycoord, xcoord = i - alpha_y[i, j], j - alpha_x[i, j]
                if round(ycoord) in y_possible_rounds and round(xcoord) in x_possible_rounds:
                    coordinates.append((j, i))  # (x, y)
                    n += 1
        #pixscale = self.pixelscale
        #plt.scatter([i[0]*pixscale for i in coordinates], [i[1]*pixscale for i in coordinates], c='r', s=1)
        #plt.scatter(coord[0]*pixscale, coord[1]*pixscale, c='b', s=1)
    
        return coordinates

    def def_angle_interpolate(self, x,y, alpha_x= None, alpha_y = None):  #(x,y) is img_guess
        """
        Interpolate the deflection angle at the image position.
        """
        alpha_x = np.array(self.alpha_map_x, dtype=np.float64)
        alpha_y = np.array(self.alpha_map_y, dtype=np.float64)
    
        dx = x - floor(x)
        dy = y - floor(y)
        top_left = np.array([alpha_x[ceil(y), floor(x)], alpha_y[ceil(y), floor(x)]]) #to match (y,x) of alpha grid
        top_right = np.array([alpha_x[ceil(y), ceil(x)], alpha_y[ceil(y), ceil(x)]])
        bottom_left = np.array([alpha_x[floor(y), floor(x)], alpha_y[floor(y), floor(x)]])
        bottom_right = np.array([alpha_x[floor(y), ceil(x)], alpha_y[floor(y), ceil(x)]])
        top = top_left * (1 - dx) + top_right * dx
        bottom = bottom_left * (1 - dx) + bottom_right * dx
        alpha = top * dy + bottom *(1 - dy)
        src_guess = np.array([x-alpha[0], y-alpha[1]])
        return src_guess, alpha
    

    def diff_interpolate (self, img_guess):
        """
        Difference between the guessed source position and the real source position.
        """
        real_src = (self.x_src, self.y_src)
        src_guess = self.def_angle_interpolate(img_guess[0],img_guess[1])[0]
        return np.sqrt((src_guess[0]-real_src[0])**2 + (src_guess[1]-real_src[1])**2)
    
    def get_image_positions(self, pixscale = None):
        """
        Get the image positions of the source.

        Parameters:
        ---------------
        x_src: The x coordinate of the source in arcsec.
        y_src: The y coordinate of the source in arcsec.
        pixscale: The pixel scale of the deflection map in arcsec/pixel.

        Returns:
        ---------------
        image_positions: The image positions of the source in arcsec.
        """
        pixscale = self.pixelscale
        #Separate the images
        coordinates = np.array(self.find_rough_def_pix()) #data in pixel

        # Apply DBSCAN clustering
        # eps and min_samples need to be chosen based on your specific data
        dbscan = DBSCAN(eps=3, min_samples=1).fit(coordinates)

        # Extract labels
        labels = dbscan.labels_

        # Separate coordinates into arrays for each image
        images = {}
        for label in set(labels):
            if label != -1:  # Ignore noise points
                images[f"Image_{label}"] = coordinates[labels == label]

        # images now contains separate arrays for each detected image
        # convert the dictionary images to list
        images = list(images.values())

        #for i in range(len(images)):
            #plt.scatter(images[i][:,0], images[i][:,1], s=0.5)
        print(f'Number of pixels: {[np.sum(len(images[i])) for i in range(len(images))]}')
        
        # Get the image positions
        plt.scatter(self.x_src* pixscale, self.y_src* pixscale, c='b')
        plt.scatter([i[0]* pixscale for i in coordinates], [i[1]* pixscale for i in coordinates], c='y', s=5)

        img = [[] for _ in range(len(images))]
       

        for i in range(len(images)):                   #pylint: disable=consider-using-enumerate
            x_max, x_min = np.max(images[i][:,0]), np.min(images[i][:,0])
            y_max, y_min = np.max(images[i][:,1]), np.min(images[i][:,1])
            img_guess = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
            pos = minimize.minimize(self.diff_interpolate, img_guess, bounds =[(x_min-2, x_max+2), (y_min-2, y_max+2)], method='L-BFGS-B', tol=1e-7) # the 2 is for wider boundary
            print(x_min* pixscale, x_max* pixscale, y_min* pixscale, y_max* pixscale, pos.x* pixscale, self.diff_interpolate(pos.x))
            plt.scatter(pos.x[0]* pixscale, pos.x[1]* pixscale, c='g', s=10, marker='x')
            img[i] = (pos.x[0]* pixscale, pos.x[1]*pixscale)

        return img