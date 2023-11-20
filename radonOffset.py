"""
Same as radonCenter, but can be used on stars that are outside the FOV
"""
import numpy as np
from scipy.interpolate import interp2d

def smoothCostFunction(costFunction, halfWidth = 0):
    """
    smoothCostFunction will smooth the function within +/- halfWidth, i.e., to replace the value with the average within +/- halfWidth pixel.
    This function can be genrally used to smooth any 2D matrix.
    Input:
        costFunction: original cost function, a matrix.
        halfWdith: the half width of the smoothing region, default = 0 pix.
    Output:
        newFunction: smoothed cost function.
    """
    if halfWidth == 0:
        return costFunction
    else:
        newFunction = np.zeros(costFunction.shape)
        rowRange = np.arange(costFunction.shape[0], dtype=int)
        colRange = np.arange(costFunction.shape[1], dtype=int)
        rangeShift = np.arange(-halfWidth, halfWidth + 0.1, dtype=int)
        for i in rowRange:
            for j in colRange:
                if np.isnan(costFunction[i, j]):
                    newFunction[i, j] = np.nan
                else:
                    surrondingNumber = (2 * halfWidth + 1) ** 2
                    avg = 0
                    for ii in (i + rangeShift):
                        for jj in (j + rangeShift):
                            if (not (ii in rowRange)) or (not (jj in colRange)) or (np.isnan(costFunction[ii, jj])):
                                surrondingNumber -= 1
                            else:
                                avg += costFunction[ii, jj]
                    newFunction[i, j] = avg * 1.0 / surrondingNumber
    return newFunction


def samplingRegion( 
        size_window : int,
        theta : float | list[float] = [45, 135],
        m : float = 0.0,
        M : float = 1.0,
        step : int = 1,
        decimals : int = 2,
        ray : bool = True,
) -> tuple[np.array, np.array]:
    """This function returns all the coordinates of the sampling region, the center of the region is (0,0)
    When applying to matrices, don't forget to SHIFT THE CENTER!
    Input:
        size_window: the radius of the sampling region. The whole region should thus have a length of 2*size_window+1.
        theta: the angle range of the sampling region, default: [45, 135] for the anti-diagonal and diagonal directions.
            measured from 0 along the x-axis and positive going counterclockwise.
        m: the minimum fraction of size_window, default: 0.2 (i.e., 20%). In this way, the saturated region can be excluded.
        M: the maximum fraction of size_window, default: 0.8 (i.e., 80%). Just in case if there's some star along the diagonals.
        step: the seperation between sampling dots (units: pixel), default value is 1pix.
        decimals: the precisoin of the sampling dots (units: pixel), default value is 0.01pix.
        ray: only half of the line?
    Output: (xs, ys)
        xs: x indices, flattened.
        ys: y indices, flattened.
    Example:
        1. If you call "xs, ys = samplingRegion(5)", you will get:
        xs: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83,  2.83, 2.12,  1.41,  0.71, -0.71, -1.41, -2.12, -2.83]
        ys: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83, -2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83]))
        2. For "radonCenter.samplingRegion(5, ray=True)", you will get:
        xs: array([ 0.71,  1.41,  2.12,  2.83, -0.71, -1.41, -2.12, -2.83])
        ys: array([ 0.71,  1.41,  2.12,  2.83,  0.71,  1.41,  2.12,  2.83])
    """
    #When there is only one angle
    if np.asarray(theta).shape == ():
        theta = [theta]
    theta = np.array(theta)

    if ray:
        zeroDegXs = np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step)
    else:
        zeroDegXs = np.append(np.arange(-int(size_window*M), -int(size_window*m) + 0.1 * step, step),
                              np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step))
    #create the column indecies if theta = 0
    zeroDegYs = np.zeros(zeroDegXs.size)
    
    xs = np.zeros((np.size(theta), np.size(zeroDegXs)))
    ys = np.zeros((np.size(theta), np.size(zeroDegXs)))
    
    for i, angle in enumerate(theta):
        degRad = np.deg2rad(angle)
        angleDegXs = np.round(zeroDegXs * np.cos(degRad), decimals = decimals)
        angleDegYs = np.round(zeroDegXs * np.sin(degRad), decimals = decimals)
        xs[i, ] = angleDegXs
        ys[i, ] = angleDegYs
    
    xs = xs.flatten()
    ys = ys.flatten()

    return xs, ys


def searchOffCenter(
        image : np.array,
        x_ctr_assign : int | float,
        y_ctr_assign : int | float,
        size_window : int,
        m : float = 0.0,
        M : float = 1.0,
        size_cost : int = 5,
        theta : list[float] = [45, 135],
        ray : bool = False,
        smooth : int = 2,
        decimals : int = 2
) -> tuple[float, float] :
    """
    Docstring goes here

    Parameters
    ----------
    image : np.array
      2-d image to find the center of
    x_ctr_assign : int | float
      initial guess for the x center (-1 for off to the left, Nx+1 for off to the right)
    y_ctr_assign : int | float
      initial guess for the y center (-1 for below, Ny+1 for above)
    size_window : int
      radial size of the search window
    m : float = 0.0
      scale size of the search window to accommodate IWA of occulting regions
    M : float = 1.0
      scale size of the search window to accommodate OWA of occulting regions
    size_cost : int = 5
      defines a square region +/- intial center within which to compute the cost function
    theta : list[float] = [45, 135]
      the angle range of the sampling region; default: [45, 135] for the anti-diagonal and diagonal directions.
    ray : bool = False
      rays go out from the center: if the theta is defined as a ray, only search that angle going out
    smooth : int = 2
      width in pixels of the smoothing window for the cost function
    decimals : int = 2
      the precision in decimal places of the centers (2 -> 0.01 precision)

    Output
    ------
    (x_cen, y_cen) : tuple of the best guess for the center

    """

    (y_len, x_len) = image.shape

    x_range = np.arange(x_len)
    y_range = np.arange(y_len)

    # this is an interpolation function to evaluate the image at subpixel coordinates
    image_interp = interp2d(x_range, y_range, image, kind = 'cubic')
    

    #The below lines create the centers of the search region
    #The cost function stores the sum of all the values in the sampling region
    # costFunction is an array covering potential values of the center guesses
    precision = 1
    x_centers = np.arange(x_ctr_assign - size_cost,
                          x_ctr_assign + size_cost + precision/10.0,
                          precision)
    x_centers = np.round(x_centers, decimals)
    y_centers = np.arange(y_ctr_assign - size_cost,
                          y_ctr_assign + size_cost + precision/10.0,
                          precision)
    y_centers = np.round(y_centers, decimals)
    costFunction = np.zeros((x_centers.shape[0], y_centers.shape[0]))

    # if the star is outside the image, you need to add this shift to search coordinates
    # find the outside edge of the search window
    # we only want to correct for the x offset
    image_offset = np.array([x_ctr_assign-size_window, 0])
    # get the pixels along the ray
    size_window = size_window - size_cost
    (xs, ys) = samplingRegion(size_window, theta, m = m, M = M, ray = ray)
    #the center of the sampling region is (0,0), so don't forget to shift the center!


    for j, x0 in enumerate(x_centers):
        for i, y0 in enumerate(y_centers):
            value = 0
            
            for x1, y1 in zip(xs, ys):
                #Shifting the center one by one, this now is the coordinate of the RAW IMAGE
                x = x0 - image_offset[0] + x1
                y = y0 - image_offset[1] + y1
                value += image_interp(x, y)
        
            costFunction[i, j] = value  #Create the cost function

    #Smooth the cost function
    costFunction = smoothCostFunction(costFunction, halfWidth = smooth)
    
    # interpolate the cost function onto the test values of the centers
    interp_costfunction = interp2d(x_centers, y_centers, costFunction, kind='cubic')

    for decimal in range(1, decimals+1):
        precision = 10**(-decimal)
        if decimal >= 2:
            size_cost = 10*precision
        x_centers_new = np.arange(x_ctr_assign - size_cost,
                                  x_ctr_assign + size_cost + precision/10.0,
                                  precision)
        x_centers_new = np.round(x_centers_new, decimals=decimal)
        y_centers_new = np.arange(y_ctr_assign - size_cost,
                                  y_ctr_assign + size_cost + precision/10.0,
                                  precision)
        y_centers_new = np.round(y_centers_new, decimals=decimal)
    
        x_cen = 0
        y_cen = 0
        maxcostfunction = 0
        value = np.zeros((y_centers_new.shape[0], x_centers_new.shape[0]))
    
        for j, x in enumerate(x_centers_new):
            for i, y in enumerate(y_centers_new):
                value[i, j] = interp_costfunction(x, y)
        
        idx = np.where(value == np.max(value))
        #Just in case when there are multile maxima, then use the average of them. 
        x_cen = np.mean(x_centers_new[idx[1]])
        y_cen = np.mean(y_centers_new[idx[0]])
        
        x_ctr_assign = x_cen
        y_ctr_assign = y_cen    
       
    x_cen = round(x_cen, decimals)
    y_cen = round(y_cen, decimals)
    return x_cen, y_cen
