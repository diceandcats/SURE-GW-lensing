"""SURE cluster lensing module."""

from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy.optimize._minimize as minimize
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.constants as const

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
        self.pixscale = pixscale
        self.size = size
        self.x_src = x_src / pixscale   #in pixel now
        self.y_src = y_src / pixscale   #in pixel now
        self.image_positions = None
        self.magnifications = None
        self.time_delays = None


    def find_rough_def_pix(self):    # result are in pixel
        """
        Find the pixels that can ray-trace back to the source position roughly.
        """
        alpha_x = self.alpha_map_x   # make sure alpha_x and alpha_y are in pixel
        alpha_y = self.alpha_map_y
        coord = (self.x_src, self.y_src)  # in pixel
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
        #pixscale = self.pixscale
        #plt.scatter([i[0]*pixscale for i in coordinates], [i[1]*pixscale for i in coordinates], c='r', s=1)
        #plt.scatter(coord[0]*pixscale, coord[1]*pixscale, c='b', s=1)
    
        return coordinates   # in pixel

    def def_angle_interpolate(self, x,y, alpha_x= None, alpha_y = None):  #(x,y) is img_guess
        """
        Interpolate the deflection angle at the image position.
        """
        alpha_x = np.array(self.alpha_map_x, dtype=np.float64)    #in pixel
        alpha_y = np.array(self.alpha_map_y, dtype=np.float64)    #in pixel
    
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
        return src_guess, alpha     # in pixel
    

    def diff_interpolate (self, img_guess):
        """
        Difference between the guessed source position and the real source position.
        """
        real_src = (self.x_src, self.y_src)   # in pixel
        src_guess = self.def_angle_interpolate(img_guess[0],img_guess[1])[0]    # in pixel
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
        pixscale = self.pixscale
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
        #print(f'Number of pixels: {[np.sum(len(images[i])) for i in range(len(images))]}')
        
        # Get the image positions
        plt.scatter(self.x_src* pixscale, self.y_src* pixscale, c='b')                                            #plot in arcsec
        plt.scatter([i[0]* pixscale for i in coordinates], [i[1]* pixscale for i in coordinates], c='y', s=5)     #plot in arcsec

        img = [[] for _ in range(len(images))]
       

        for i in range(len(images)):                   #pylint: disable=consider-using-enumerate
            x_max, x_min = np.max(images[i][:,0]), np.min(images[i][:,0])
            y_max, y_min = np.max(images[i][:,1]), np.min(images[i][:,1])
            img_guess = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
            pos = minimize.minimize(self.diff_interpolate, img_guess, bounds =[(x_min-2, x_max+2), (y_min-2, y_max+2)], method='L-BFGS-B', tol=1e-7) # the 2 is for wider boundary
            #print(x_min* pixscale, x_max* pixscale, y_min* pixscale, y_max* pixscale, pos.x* pixscale, self.diff_interpolate(pos.x))
            plt.scatter(pos.x[0]* pixscale, pos.x[1]* pixscale, c='g', s=10, marker='x')
            img[i] = (pos.x[0]* pixscale, pos.x[1]*pixscale)

        return img              # in arcsec


    def get_magnifications(self, h = 1e-9):
        """
        Get the magnifications of the images.

        Returns:
        ---------------
        magnifications: The magnifications of the images.
        """
        def partial_derivative(func, var, point): 
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(args)

            #print(wraps(point[var]+h), wraps(point[var]-h))

            return lambda x: (wraps(x+h) - wraps(x-h))/(2*h) # central difference diff fct

        def alpha(t):
            alpha = self.def_angle_interpolate(t[0], t[1])[1]
            a = float(f"{alpha[0]:.12f}")
            b = float(f"{alpha[1]:.12f}")
            return np.array([a, b])

        theta = np.array(self.get_image_positions())/self.pixscale    #in pixel
        magnification = []

        for theta in enumerate(theta):
            dalpha1_dtheta1 = partial_derivative(lambda t: alpha(t)[0], 0, theta[1])(theta[1][0])
            dalpha1_dtheta2 = partial_derivative(lambda t: alpha(t)[0], 1, theta[1])(theta[1][1])
            dalpha2_dtheta1 = partial_derivative(lambda t: alpha(t)[1], 0, theta[1])(theta[1][0])
            dalpha2_dtheta2 = partial_derivative(lambda t: alpha(t)[1], 1, theta[1])(theta[1][1])
            #print(dalpha1_dtheta1, dalpha1_dtheta2, dalpha2_dtheta1, dalpha2_dtheta2)


            # Construct the magnification tensor
            a = np.array([
                [1 - dalpha1_dtheta1, -dalpha1_dtheta2],
                [-dalpha2_dtheta1, 1 - dalpha2_dtheta2]
            ])

            # Calculate magnification
            magnification.append( 1 / np.linalg.det(a))

        return magnification
    
    def get_time_delays(self):
        """
        Get the time delays of the images.

        Returns:
        ---------------
        time_delays: The time delays of the images in days.
        """
        
        theta = self.get_image_positions()     #in arcsec
        beta = np.array([self.x_src * self.pixscale, self.y_src * self.pixscale])   #in arcsec
        data_psi_arcsec = self.lens_potential_map  #in arcsec^2

        def psi_interpolate(x,y, psi = data_psi_arcsec):  #(x,y) is img in arcsec 
            x = x/self.pixscale
            y = y/self.pixscale
            dx = x - floor(x)
            dy = y - floor(y)
            top_left = np.array(psi[ceil(y), floor(x)]) #to match (y,x) of alpha grid
            top_right = np.array(psi[ceil(y), ceil(x)])
            bottom_left = np.array(psi[floor(y), floor(x)])
            bottom_right = np.array(psi[floor(y), ceil(x)])
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            psi = top * dy + bottom *(1 - dy)
            return psi

        def fermat_potential(theta, beta):
            return 0.5 * (np.linalg.norm(theta - beta)**2) - psi_interpolate(theta[0], theta[1])
            

        #for i in range(len(theta)):     #pylint: disable=consider-using-enumerate
            #print(f"Interpolation Fermat potential at {theta[i]}: {fermat_potential(np.array(theta[i]), beta)}")

        # time delay by diff of fermat potentials and scale it by time-delay distance
        dt = []
        for i in range(len(theta)):  #pylint: disable=consider-using-enumerate
            dt.append(fermat_potential(np.array(theta[i]), beta) - fermat_potential(np.array(theta[0]), beta))
            #print(f"demensionless time delay at {theta[i]}: {dt[i]}")

        # Redshifts
        z_L = 0.5
        z_S = 1.0

        # Calculate distances
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_L = cosmo.angular_diameter_distance(z_L)
        D_S = cosmo.angular_diameter_distance(z_S)
        D_LS = cosmo.angular_diameter_distance_z1z2(z_L, z_S)
        #print(D_LS)
        time_delay_distance = (1 + z_L) * D_L * D_S / D_LS * const.Mpc
        #print(f"Time-delay distance: {time_delay_distance.value}")
        dt_days = np.array(dt) * time_delay_distance.value / const.c / const.day_s * const.arcsec ** 2
        print(f"Numerical time delay in days: {dt_days} days")
        return dt_days


    

        