"""SURE cluster lensing module."""

from math import ceil, floor
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy.optimize._minimize as minimize
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.constants as const
import pandas as pd

# pylint: disable=C0103
class ClusterLensing:
    """
    Class to get the lensing properties of a cluster by deflection and lens potential map of the cluster.
    """

    def __init__(self, alpha_map_x, alpha_map_y, lens_potential_map, z_l , z_s, pixscale, size, diff_z = False):
        """
        Parameters:
        ---------------
        alpha_map_x: The deflection map in x direction in arcsec.
        alpha_map_y: The deflection map in y direction in arcsec
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
        self.image_positions = None
        self.magnifications = None
        self.time_delays = None
        self.diff_z = diff_z

        if diff_z:
            self.D_S1, self.D_S2, self.D_LS1, self.D_LS2 = self.scaling()

    def scaling(self):
        """
        Scale the deflection and lens potential maps.
        """
        # Redshifts
        z_L = self.z_l
        z_S = self.z_s

        # Calculate distances
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_S1 = cosmo.angular_diameter_distance(1.0)
        D_S2 = cosmo.angular_diameter_distance(z_S)
        D_LS1 = cosmo.angular_diameter_distance_z1z2(z_L, 1.0)
        D_LS2 = cosmo.angular_diameter_distance_z1z2(z_L, z_S)

        # Scale deflection map
        scal = (D_LS1 * D_S2)/(D_LS2 * D_S1)
        self.alpha_map_x *= scal
        self.alpha_map_y *= scal

        return D_S1, D_S2, D_LS1, D_LS2

    def find_rough_def_pix(self, x_src, y_src):    # result are in pixel
        """
        Find the pixels that can ray-trace back to the source position roughly.
        """
        alpha_x = self.alpha_map_x   # make sure alpha_x and alpha_y are in pixel
        alpha_y = self.alpha_map_y
        coord = (x_src, y_src)  # in pixel
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
        return src_guess, alpha


    def diff_interpolate (self, img_guess, x_src, y_src):
        """
        Difference between the guessed source position and the real source position.
        """
        real_src = (x_src, y_src)   # in pixel
        src_guess = self.def_angle_interpolate(img_guess[0],img_guess[1])[0]    # in pixel
        return np.sqrt((src_guess[0]-real_src[0])**2 + (src_guess[1]-real_src[1])**2)

    def clustering(self, x_src, y_src):
        """
        Cluster the image positions.
        """
        coordinates = np.array(self.find_rough_def_pix(x_src, y_src))
        if len(coordinates) == 0:
            return []
        dbscan = DBSCAN(eps=3, min_samples=1).fit(coordinates)
        labels = dbscan.labels_
        images = {}
        for label in set(labels):
            if label != -1:
                images[f"Image_{label}"] = coordinates[labels == label]
        images = list(images.values())
        return images


    def get_image_positions(self, x_src, y_src, pixscale = None):
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
        x_src = x_src / pixscale
        y_src = y_src / pixscale
        images = self.clustering(x_src, y_src)

        #for i in range(len(images)):
            #plt.scatter(images[i][:,0], images[i][:,1], s=0.5)
        #print(f'Number of pixels: {[np.sum(len(images[i])) for i in range(len(images))]}')

        # Get the image positions
        #plt.scatter(self.x_src* pixscale, self.y_src* pixscale, c='b')                                            #plot in arcsec
        #plt.scatter([i[0]* pixscale for i in coordinates], [i[1]* pixscale for i in coordinates], c='y', s=5)     #plot in arcsec

        img = [[] for _ in range(len(images))]

        def wrap_diff_interpolate(img_guess):
            return self.diff_interpolate(img_guess, x_src, y_src)


        for i, image in enumerate(images):
            x_max, x_min = np.max(image[:,0]), np.min(image[:,0])
            y_max, y_min = np.max(image[:,1]), np.min(image[:,1])
            img_guess = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
            pos = minimize.minimize(wrap_diff_interpolate, img_guess, bounds =[(x_min-1.5, x_max+1.5), (y_min-1.5, y_max+1.5)], method='L-BFGS-B', tol=1e-9) # the 2 is for wider boundary
            #print(x_min* pixscale, x_max* pixscale, y_min* pixscale, y_max* pixscale, pos.x* pixscale, self.diff_interpolate(pos.x))
            #plt.scatter(pos.x[0]* pixscale, pos.x[1]* pixscale, c='g', s=10, marker='x')
            img[i] = (pos.x[0]* pixscale, pos.x[1]*pixscale)   # in arcsec
            
        img = sorted(img, key=lambda x: x[0])
        
        return img              # in arcsec

    def partial_derivative(self, func, var, point, h = 1e-9):
        """
        Calculate the partial derivative of a function.
        """
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)

        #print(wraps(point[var]+h), wraps(point[var]-h))

        return lambda x: (wraps(x+h) - wraps(x-h))/(2*h) # central difference diff fct

    def get_magnifications(self, theta):
        """
        Get the magnifications of the images.

        Parameters:
        ---------------
        theta : tuple of image position(x,y) in arcsec

        Returns:
        ---------------
        magnifications: The magnifications of the images.
        """

        def alpha(t):
            alpha = self.def_angle_interpolate(t[0], t[1])[1]
            return alpha
        theta_arcsec = theta           # in tuple
        theta = np.array(theta_arcsec)/self.pixscale    #in pixel
        
        magnification = []

        for theta in enumerate(theta):
            dalpha1_dtheta1 = self.partial_derivative(lambda t: alpha(t)[0], 0, theta[1])(theta[1][0])
            dalpha1_dtheta2 = self.partial_derivative(lambda t: alpha(t)[0], 1, theta[1])(theta[1][1])
            dalpha2_dtheta1 = self.partial_derivative(lambda t: alpha(t)[1], 0, theta[1])(theta[1][0])
            dalpha2_dtheta2 = self.partial_derivative(lambda t: alpha(t)[1], 1, theta[1])(theta[1][1])
            #print(dalpha1_dtheta1, dalpha1_dtheta2, dalpha2_dtheta1, dalpha2_dtheta2)


            # Construct the magnification tensor
            a = np.array([
                [1 - dalpha1_dtheta1, -dalpha1_dtheta2],
                [-dalpha2_dtheta1, 1 - dalpha2_dtheta2]
            ])

            # Calculate magnification
            magnification.append( 1 / np.linalg.det(a))

        # using dataframe to store the magnification

        #for dataframe display
        #data_mag=[]
        #for i , (x,y) in enumerate(theta_arcsec):
            #mag = magnification[i]
            #data_mag.append({'x': x, 'y': y, 'magnification': mag})
        
        #table = pd.DataFrame(data_mag)
        #pd.options.display.float_format = '{:.12f}'.format
        #if table.empty:
            #return []
        #table = table.sort_values(by=['x']).reset_index(drop=True)
        
        return magnification
    

    def psi_interpolate(self, x,y):  #(x,y) is img in arcsec 
        """
        Interpolate the lens potential at the image position.
        """
        psi = self.lens_potential_map   #in arcsec^2
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

    def fermat_potential(self, theta, beta):
        """Get the Fermat potential at the image position."""
        return 0.5 * (np.linalg.norm(theta - beta)**2) - self.psi_interpolate(theta[0], theta[1])

    def mp_fermat_potential(self, theta, beta):
        """Get the Fermat potential at the image position for mp case."""
        factor = self.D_S2 * self.D_LS1 / self.D_LS2 / self.D_S1
        return 0.5 * (np.linalg.norm(theta - beta)**2) - factor *self.psi_interpolate(theta[0], theta[1])

    def get_time_delays(self, x_src, y_src, theta):
        """
        Get the time delays of the images.

        Returns:
        ---------------
        time_delays: The time delays of the images in days.
        """
        #theta = self.get_image_positions()     #in arcsec
        beta = np.array([x_src, y_src])   #in arcsec

        # Redshifts
        z_L = self.z_l
        z_S = self.z_s

        # Calculate distances
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_L = cosmo.angular_diameter_distance(z_L)
        D_S = cosmo.angular_diameter_distance(z_S)
        D_LS = cosmo.angular_diameter_distance_z1z2(z_L, z_S)
        #print(D_LS)
        time_delay_distance = (1 + z_L) * D_L * D_S / D_LS * const.Mpc

        if self.diff_z:
            fermat_potential = [self.mp_fermat_potential(np.array(pos), beta) for pos in theta]
            if len(fermat_potential) == 0:
                return []
            min_fermat = min(fermat_potential)
            dt = [fermat_potential[i] - min_fermat for i in range(len(fermat_potential))]
            dt_days = np.array(dt) * time_delay_distance.value / const.c / const.day_s * const.arcsec ** 2
            data = {
                'theta_x': [pos[0] for pos in theta],
                'theta_y': [pos[1] for pos in theta],
                'd_fermat': dt,
                'delta_t(days)': dt_days
            }
            df = pd.DataFrame(data)
            df_sorted = df.sort_values(by='d_fermat').reset_index(drop=True)
        
            #print(f"Time-delay distance: {time_delay_distance.value}")
            #print(f"Numerical time delay in days: {dt_days} days")
            return df_sorted, dt_days

        else:
            #for i in range(len(theta)):     #pylint: disable=consider-using-enumerate
                #print(f"Interpolation Fermat potential at {theta[i]}: {fermat_potential(np.array(theta[i]), beta)}")

            # time delay by diff of fermat potentials
        
            fermat_potential = [self.fermat_potential(np.array(pos), beta) for pos in theta]
            min_fermat = min(fermat_potential)
            dt = [fermat_potential[i] - min_fermat for i in range(len(fermat_potential))]
            
            dt_days = np.array(dt) * time_delay_distance.value / const.c / const.day_s * const.arcsec ** 2
            data = {
                'theta_x': [pos[0] for pos in theta],
                'theta_y': [pos[1] for pos in theta],
                'd_fermat': dt,
                'delta_t(days)': dt_days
            }
            df = pd.DataFrame(data)
            df_sorted = df.sort_values(by='d_fermat').reset_index(drop=True)
        
            #print(f"Time-delay distance: {time_delay_distance.value}")
            #print(f"Numerical time delay in days: {dt_days} days")
            return df_sorted
    

    def all_dt(self, x_src, y_src, theta):
        '''
        Get all the time delays of the images.

        Returns:
        ---------------
        time_delays: The time delays of the images in days.
        '''
        beta = np.array([x_src, y_src])   #in arcsec

        # Redshifts
        z_L = self.z_l
        z_S = self.z_s

        # Calculate distances
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_L = cosmo.angular_diameter_distance(z_L)
        D_S = cosmo.angular_diameter_distance(z_S)
        D_LS = cosmo.angular_diameter_distance_z1z2(z_L, z_S)
        #print(D_LS)
        time_delay_distance = (1 + z_L) * D_L * D_S / D_LS * const.Mpc

        if self.diff_z:
            fermat_potential = [self.mp_fermat_potential(np.array(pos), beta) for pos in theta]
            if len(fermat_potential) == 0:
                return []
            all_dt = []
            for i in range(len(fermat_potential)):          #pylint: disable=consider-using-enumerate
                for j in range(i+1, len(fermat_potential)):
                    diff = abs(fermat_potential[i] - fermat_potential[j])
                    all_dt.append(diff)
            
            dt_days = np.array(all_dt) * time_delay_distance.value / const.c / const.day_s * const.arcsec ** 2
        
            #print(f"Time-delay distance: {time_delay_distance.value}")
            #print(f"Numerical time delay in days: {dt_days} days")
            return dt_days



        