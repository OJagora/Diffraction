import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

N = 1000  # image size

# ======= Parameters ======= #

D = 0.102 #m
d = 0.660 #m
L = 0.001 #m

#d = 57.6
#D = 2.4
#L = 0.036

#======= Diffraction Masks ============#

#diafragm
def f1(x,y):
    return (np.sqrt(x**2 + y**2)) <= D/2

#telescope
def f2(x,y):
    c1 = np.sqrt(x**2 + y**2) <= D/2
    c2 = np.sqrt(x**2 + y**2) >= 0.2 * D/2
    c3 = abs(x) >= 0.05 * D/2
    c4 = abs(y) >= 0.05 * D/2
    
    return (c1*c2*c3*c4)

#square
def f3(x,y):
    c1 = abs(x) >= D
    c2 = abs(y) >= D
    return c1*c2

#slits
def f4(x,y):
    c1 = abs(y) >= D*0.6
    c2 = abs(x) >= D*1.3
    c3 = abs(x) < D*1.4
    return c1*c2*c3

#====== fft Calculation =========#


def fft2(A):
    """
    Calculate the diffraction pattern of the image A

    Parameters
    ----------
    A : 2D array
        The image to calculate the diffraction pattern of.

    Returns
    -------
    I : 2D array
        The diffraction pattern of the image A.
    """
    U = np.fft.fft2(A)            #TFD2D of the diaphragm
    U = np.fft.fftshift(U)        #Centred on the zero frequency
    I = np.square(np.abs(U))      #Intensity of the diffraction pattern (square of the modulus of the FFT)
    return(I)

def echant(lamb, f):
    """
    Sample the image A with a grid of size lamb

    Parameters
    ----------
    lamb : float
        The size of the grid to sample the image with.

    f : function
        The function representing the mask to sample the image with.

    Returns
    -------
    A : 2D array
        The sampled image.
    """
    A = np.zeros((N,N))
    xi = xe(lamb*10**(-9))
    for x in range(-int(N/2),int(N/2)) :
        for y in range(-int(N/2),int(N/2)) :

            dx = x*xi
            dy = y*xi

            A[x,y] = f(dx,dy)

    return A

def filtre(I,p):
    """
    Filter the values of the image I

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
    for x in range(-int(N/2),int(N/2)) :
        for y in range(-int(N/2),int(N/2)) :
            if I[x,y] > p*m :
                I[x,y] = np.log(I[x,y])
            else :
                I[x,y] = 0
    return I

#======= Polychromatism =============#

def xe(lamb) :
    return (lamb*d)/L

def wave2rgb(wave):
    """
    Convert a wavelength to an RGB color.
    """
    gamma = 0.8
    intensity_max = 1
 
    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0
 
    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0
 
    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow (c * factor, gamma)
 
    return f(red), f(green), f(blue)



def tocolor(lamb) :
    """
    Calculate the diffraction pattern of the image A in color
    """
    R,G,B = wave2rgb(lamb)                      #code RGB associe a la longueur
    A = echant(lamb, f2)                            #echantillonnage
    I = fft2(A)                                 #calcul de la figure de diffraction en intensite
    I = filtre(I,0.001)                               #filtre des donnees

    return I,R,G,B

def polychroma(n):
    """
    Iterate the process on n wavelengths
    """
    Z = np.zeros((N,N))
    Itot =  np.dstack((Z,Z,Z))
    LAMB = np.linspace(380, 780, n)             #choix de n longueurs d'ondes entre 380 et 780nm (dans le domaine du visible)
    for lamb in LAMB :                          #boucle et somme des images
        print("Itération pour lambda = ", lamb, " nm")
        I,R,G,B = tocolor(lamb)
        Itot = Itot + np.dstack((R*I,G*I,B*I))

        Ir = np.dstack((R*I,G*I,B*I))
        It = np.interp(Itot, (Itot.min(),Itot.max()),(0,1))

    Itot = np.interp(Itot, (Itot.min(),Itot.max()),(0,1))
    Itot = (Itot*255).astype(np.uint8)
    Itot = Itot[400:600,400:600]
    Itot = Image.fromarray(Itot, 'RGB')
    return Itot
    
#====== affichage ==========#
k = 100
Itot = polychroma(k)

#path = ""
#Itot = (Itot*255).astype(np.uint8)
#Itot = Image.fromarray(Itot, 'RGB')
#Itot.save(path)

plt.imshow(Itot)
plt.title("Diffraction Pattern of a custom diafragm for " + str(k) + " wavelengths of white light")
plt.show()




