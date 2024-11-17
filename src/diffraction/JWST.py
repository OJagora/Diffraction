import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image

# ================================================= #
# ======= Diffraction Pattern of the JWST ========= #
# ================================================= #

# ------ Load the image of the JWST ------ #

masque = Image.open("././resources/image_construct/"+
                    "JWST_Nircam_alignment_selfie_labeled.jpg")

masque = masque.convert('L')
masque = np.array(masque)

masque = resize(masque, (1000, 1000), mode = 'reflect')

masque = masque > 0.6

# ------ Parameters ------ #

N = 1000  # size of the image
    
D = int(873 / 6.5)    #D is the caracteristic size of the image
                      # it is the ratio between the size of the image and the size of the figure (in pixels per meter)
                      # the size of the figure is 873 pixels for a 6.5 meters diameter

D = 134.3

d = 131.4 #m
L = 0.036 #m

# ======= Calculation of the FFT ========= #

def fft2(A):
    """
    This function calculates the diffraction pattern of the image A

    Parameters
    ----------
    A : 2D array
        The image to calculate the diffraction pattern of.
    
    Returns
    -------
    I : 2D array
        The diffraction pattern of the image A.
    """
    U = np.fft.fft2(A)            # FFT2 of the diafragm
    U = np.fft.fftshift(U)        # Center on the zero frequency
    I = np.square(np.abs(U))      # Intensity of the diffraction pattern (square of the modulus of the FFT)
    return(I)

def echant(lamb):
    """
    This function samples the image A with a grid of size lamb

    Parameters
    ----------
    lamb : float
        The size of the grid to sample the image with.

    Returns
    -------
    A : 2D array
        The sampled image.
    """
    A = np.zeros((N, N))
    xi = xe(lamb * 10**(-9))
    for x in range(-int(N / 2), int(N / 2)) :
        for y in range(-int(N / 2), int(N / 2)) :

            dx = x * xi
            dy = y * xi

            nx = int(dx * D)
            ny = int(dy * D)

            if((abs(nx) <= N / 2) and (abs(ny) <= N / 2)):
                A[x, y] = masque[nx, ny]
            else :
                A[x, y] = 0

    return A

def filtre(I, p):
    """
    This function filters the values of the image I

    Parameters
    ----------
    I : 2D array
        The image to filter.
    p : float
        The threshold value.

    Returns
    -------
    I : 2D array
        The filtered image.
    """
    m = np.max(I)
    for x in range(-int(N / 2), int(N / 2)) :
        for y in range(-int(N / 2), int(N / 2)) :
            if I[x, y] > p * m :
                I[x, y] = np.log(I[x, y])
            else :
                I[x, y] = 0
    return I

#======= Polychromatism =============#

def xe(lamb) :
    """
    Calculation of the step size for the wavelength

    Parameters
    ----------
    lamb : float
        The wavelength.

    Returns
    -------
    float
        The step size.
    """
    return (lamb * d) / L

#construction de l'image de diffraction en couleur
def tocolor(lamb) :
    """
    This function calculates the diffraction pattern of the image A in color

    Parameters
    ----------
    lamb : float
        The wavelength. 

    Returns
    -------
    I : 2D array
        The diffraction pattern of the image A in color.
    """

    A = echant(lamb)    # sampling
    I = fft2(A)         # calculation of the diffraction pattern intensity
    #I = filtre(I,0.001)                           #filtering   
    return I

def polychroma(n):
    """
    This function iterates the process on n wavelengths

    Parameters
    ----------
    n : int
        The number of wavelengths.

    Returns
    -------
    Itot : 2D array
        The diffraction pattern of the image A in color.
    """
    Itot = np.zeros((N, N))             
    LAMB = np.linspace(0.6 * 10**3, 28 * 10**3, n)             #choice of n wavelengths in the JWST range
    for lamb in LAMB :                          #iteration on the wavelengths and sum of the diffraction patterns
        print("Itération pour lambda = ", lamb, " nm")
        I = tocolor(lamb)
        Itot = I + Itot
    Itot = np.interp(Itot, (Itot.min(), I.max()), (0, 1))
    return Itot

#====== Drawing ==========#
k = 10
Itot = polychroma(k)
plt.style.use('dark_background')

plt.title("Diffraction Pattern of the JWST for " + str(k) + " wavelengths")

Im = (Itot*255).astype(np.uint8)
Im = Image.fromarray(Im, 'L')
plt.imshow(Itot, cmap = 'hot')
plt.show()
