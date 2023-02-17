import PIL
from PIL import Image
from PIL import ImageGrab
import numpy as np
import time
import math
import random
import cv2
import os

def PIL_save(img: Image.Image, filename: str, format: str = "PNG") -> None:
    """ ### Saves the PIL.Image.Image object as an image file (PNG by default) with the given name
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `var = PIL_load(img) ` \n
        `var[20, 10] = (255, 0, 0) `\n
        `PIL_save(img, "image_edited1")  ` \n
        `PIL_save(img, "image_edited2.jpg")` \n
        `PIL_save(img, "image_edited3", "JPG")` \n
        `This will change the pixel at (20, 10) to red and save the image as image_edited*.png ` \n
    
    You can check the [PIL documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html) for more info 
    """
    if filename.endswith(".png"):
        img.save(filename, "PNG")
    elif filename.endswith(".jpg"):
        img.save(filename, "JPG")
    else:
        img.save(filename + "." + format.lower(), format)

def PIL_open(filename: str) -> Image.Image:
    """ ### Opens an image file using PIL (Python Image Library)
    Returns a PIL.Image.Image object
    
    Usage: \n
        `img1 = PIL_open("image.png")` \n
        `img2 = PIL_open("image")` \n
        
    You can check the [PIL documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html) for more info 
    """
    try:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            res = Image.open(filename)
        else:
            try:
                res = Image.open(filename + ".png")
            except FileNotFoundError:
                res = Image.open(filename + ".jpg")
        return res
    except FileNotFoundError:
        print("File not found.")
        
def PIL_load(image: Image.Image) -> Image.Image.load:
    """ ### Loads the PIL.Image.Image object that you can create using the PIL_open() function, allowing you to iterate through and edit the pixels in it
    Returns a PIL.Image.Image.load object
    \n
    Usage: \n
        `img = PIL_open("image.png")` \n
        `var = PIL_load(img) ` \n
        `var[20, 10] = (255, 0, 0) `\n
        `PIL_save(img, "image_edited")  ` \n
        ` # This will change the pixel at (20, 10) to red and save the image as image_edited.png ` \n
        
    You can check the [PIL documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html) for more info 
    """
    return image.load()



def distance(c1: tuple, c2: tuple) -> float:
    """ ### Returns the distance between two colors in RGB space
    Returns a float
    Usage: \n
        `distance((30, 0, 0), (0, 40, 0)) == 50` \n
        `img = PIL_open("image.png")` \n
        `img_colors = img.convert("RGB") ` \n
        `distance(img_colors[0, 0], img_colors[1, 1])` \n
    """
    if c1 == c2:
        return 0
    return math.sqrt(((c2[0]-c1[0])**2+(c2[1] - c1[1])**2+(c2[2] - c1[2])**2))

def matrix_create(img: Image.Image) -> np.ndarray:
    """ ### Returns a numpy array of the image for faster processing
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
    """
    img_colors = img.convert('RGB')
    loaded = PIL_load(img_colors)
    width = img.size[0]
    height = img.size[1]
    res = np.zeros((height,width), dtype=tuple)
    for x in range(width):
        for y in range(height):
            res[y, x] = loaded[x, y]
    return res



def around(matrix: np.ndarray, x: int, y: int) -> list:
    """ ### Checks around matrix[y, x] and returns a list of the color values of the pixels around it
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `around(matrix, 10, 10)` \n
    """
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            try:
                neighbors.append(matrix[y+i, x+j])
            except IndexError:
                pass
    return neighbors

def check_contrast(matrix: np.ndarray, x: int, y: int) -> float:
    """ ### Returns the contrast value of the pixel at matrix[y, x]
    Calculated as the sum of the distance between the pixel and its neighbors
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `check_contrast(matrix, 10, 10)` \n
    """
    if type(matrix[0, 0]) == float:
        return matrix
    neighbors = around(matrix, x, y)
    res = 0.0
    for i in neighbors:
        res += distance(matrix[y, x], i)
    return res

def create_contrast_matrix(matrix: np.ndarray) -> np.ndarray:
    """ ### Returns a matrix of the contrast values of the pixels in the given matrix
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `contrast_matrix = create_contrast_matrix(matrix)` \n
    """
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = np.zeros((height,width))
    for x in range(width):
        for y in range(height):
            res[y, x] = check_contrast(matrix, x, y)
    return res

def Simplify(input: str | np.ndarray | Image.Image, threshold: int) -> Image.Image:
    """ ### Applies the [Simplify](https://www.github.com/EgeEken/Simplify) algorithm to the given image)
    This results in a image with only black and white pixels, black pixels being the ones in high contrast points
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `contrast_matrix = create_contrast_matrix(matrix)` \n
        `simplified_400 = Simplify(400, "image.png")` \n
        `simplified_300 = Simplify(300, img)` \n
        `simplified_200 = Simplify(200, matrix)` \n	
        `simplified_100 = Simplify(100, contrast_matrix)` \n
    """
    if type(input) == str:
        contrastmatrix = create_contrast_matrix(matrix_create(PIL_open(input)))
    elif type(input) == Image.Image:
        contrastmatrix = create_contrast_matrix(matrix_create(input))
    elif type(input) == np.ndarray:
        contrastmatrix = create_contrast_matrix(input)
    width = contrastmatrix.shape[1]
    height = contrastmatrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            if contrastmatrix[y, x] >= threshold:
                res_pixels[x, y] = (0, 0, 0)
    return res



def Brighten_color(coef: float, c: tuple) -> tuple:
    if coef == 0:
        return c
    elif coef > 1:
        coef = 1
    elif coef < -1:
        coef = -1
    r, g, b = c

    if coef < 0:
        return (int(r + r * coef),
                int(g + g * coef),
                int(b + b * coef))
        
    return (int(r + (255 - r) * coef ),
            int(g + (255 - g) * coef ),
            int(b + (255 - b) * coef ))

def Brighten(coef: float, input: str | np.ndarray | Image.Image) -> Image.Image:
    """### Brightens or darkens the image using the [Brighten](https://www.github.com/EgeEken/Brighten) function
    
    A coefficient of -1 would make the image completely black, while a coefficient of 1 would make the image completely white
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `brightened_50 = Brighten(0.5, "image.png")` \n
        `same_0 = Brighten(0, img)` \n
        `darkened_50 = Brighten(-0.5, matrix)` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    if coef > 1:
        print("Warning: Coefficient is greater than 1, setting it to 1")
    if coef < -1:
        print("Warning: Coefficient is less than -1, setting it to -1")
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            res_pixels[x, y] = Brighten_color(coef, matrix[y, x])
    return res

def Saturate_color(coef: float, c: tuple) -> tuple:
    if coef == 0:
        return c
    if coef > 1:
        coef = 1
    if coef < -1:
        coef = -1
    r, g, b = c
    cmax = max(c)
    
    if coef < 0:
        return (int(r + (r - cmax) * coef),
                int(g + (g - cmax) * coef),
                int(b + (b - cmax) * coef))
    
    cmin = min(c)

    if r == cmax:
        rmin = cmax
    elif r == cmin:
        rmin = 0
    else:
        rmin = 2*r - cmax
    
    if g == cmax:
        gmin = cmax
    elif g == cmin:
        gmin = 0
    else:
        gmin = 2*g - cmax

    if b == cmax:
        bmin = cmax
    elif b == cmin:
        bmin = 0
    else:
        bmin = 2*b - cmax

    rchange = (r - rmin) * coef
    gchange = (g - gmin) * coef
    bchange = (b - bmin) * coef
    return (int(r - rchange), int(g - gchange), int(b - bchange))

def Saturate(coef: float, input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Saturates or desaturates the given color using the [Saturate](https://www.github.com/EgeEken/Saturate) function
    
    A coefficient of -1 would make the image black and white, and a coefficient of 1 would maximize the saturation
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `saturated_50 = Saturate(matrix, 0.5)` \n
        `desaturated_50 = Saturate(matrix, -0.5)` \n    
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    if coef > 1:
        print("Warning: Coefficient is greater than 1, setting it to 1")
    if coef < -1:
        print("Warning: Coefficient is less than -1, setting it to -1")
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            res_pixels[x, y] = Saturate_color(coef, matrix[y, x])
    return res



def BnW(input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Converts the image to black and white
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `black_and_white1 = BnW_mean("image.png")` \n
        `black_and_white2 = BnW_mean(matrix)` \n
        `black_and_white3 = BnW_mean(img)` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            r, g, b = matrix[y, x]
            mean = (r + g + b) // 3
            res_pixels[x, y] = (mean, mean, mean)
    return res

def Normalize(input: str | np.ndarray | Image.Image) -> np.ndarray:
    """ ### Converts the image to black and white, returns a matrix of floats between 0 and 1, where 0 is black and 1 is white
    Primary intended use case is for neural networks
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `normalized = Normalize(img)` \n
        `print(normalized[0, 0]) #for the value of the top left pixel between 0 and 1` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    bnw = BnW(matrix)
    loaded = PIL_load(bnw)
    width = matrix.shape[0]
    height = matrix.shape[1]
    res = np.zeros((width, height), dtype=float)
    for x in range(width):
        for y in range(height):
            res[x, y] = loaded[y, x][0] / 255
    return res

def deNormalize(matrix: np.ndarray) -> Image.Image:
    """ ### Converts a normalized matrix back to an image
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `normalized = Normalize(img)` \n
        `denormalized = deNormalize(normalized)` \n
        `PIL_save(denormalized, "image_bnw.png")` \n
    """
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            rgb = int(float(matrix[y, x]) * 255)
            res_pixels[x, y] = (rgb, rgb, rgb)
    return res

def closest_normal(c: float, normalset: set | np.ndarray) -> float:
    """ ### Finds the closest color to the given color in the given color set, both using normalized values between 1 and 0"""
    if c in normalset:
        return c
    if type(normalset) == set:
        normalset = list(normalset)
    array = np.array(normalset)
    idx = (np.abs(array - c)).argmin()
    return array[idx]

def simpler_Normalize(input: str | Image.Image | np.ndarray, colorcount: int) -> np.ndarray:
    """ ### Returns a matrix of floats between 0 and 1, where 0 is black and 1 is white
    Primary intended use case is for neural networks
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `normalized = simpler_Normalize(img, 4)` \n
        `print(normalized[0, 0]) #for the value of the top left pixel, will be either 0.0, 0.25, 0.5, 0.75 or 1.0` \n
    """
    matrix = Normalize(input)
    colorset = set(np.arange(0, 1, 1 / (colorcount - 1))) | {1}
    res = np.zeros(matrix.shape, dtype=float)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            res[y, x] = closest_normal(matrix[y, x], colorset)
    return res

def array2line(input: np.ndarray | Image.Image | str) -> str:
    """ ### Returns a line of text representing the normalized version of the given image
    Primary intended use case is for neural networks"""
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    res = ""
    for i in matrix:
        for j in i:
            res += str(j)
    return res
    
    
    
def colorset_create(input: np.ndarray | Image.Image | str) -> set:
    """ ### Creates a set of all the different colors in an image matrix
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `colorset = colorset_create(matrix)` \n
        `print(len(colorset)) #for the number of colors` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    res = set()
    for i in matrix:
        res |= set(i)
    return res

def closest_color(c: tuple, colorset: set | list | np.ndarray) -> tuple:
    """ ### Finds the closest color to the given color in the given color set"""
    if c in colorset:
        return c
    mindist = 600
    for color in colorset:
        dist = distance(c, color)
        if dist < mindist:
            mindist = dist
            mincolor = color
    return mincolor

def closest_color_strict(c: tuple, colorset: set | list | np.ndarray) -> tuple:
    """ ### Finds the closest color to the given color in the given color set, but only if the colors are not equal"""
    mindist = 600
    for color in colorset:
        dist = distance(c, color)
        if dist < mindist and dist != 0:
            mindist = dist
            mincolor = color
    return mincolor
    
def SimplifyColorV4(colorcount: int, input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [SimplifyColorV4](https://www.github.com/EgeEken/Simplify-Color) algorithm to the given image
    This program recreates the input image using the given number of colors.
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV4(100, "image.png")` \n
        `simplified_50 = SimplifyColorV4(50, img)` \n
        `simplified_10 = SimplifyColorV4(10, matrix)` \n
        `simplified_5 = SimplifyColorV4(5, simplified_10)` \n
        `PIL_save(simplified_5, "simplified_5")` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    if colorcount < 1:
        print("Warning: Color count is smaller than 1, setting it to 1")
        colorcount = 1
    colorset = set(random.sample(list(colorset_create(matrix)), colorcount))
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            res_pixels[x, y] = closest_color(matrix[y, x], colorset)
    return res

def Quantize(colorcount: int, input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [SimplifyColor V4](https://www.github.com/EgeEken/Simplify-Color) algorithm to the given image
    This program recreates the input image using the given number of colors, colors are sampled randomly from the image.
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV4(100, "image.png")` \n
        `simplified_50 = SimplifyColorV4(50, img)` \n
        `simplified_10 = SimplifyColorV4(10, matrix)` \n
        `simplified_5 = SimplifyColorV4(5, simplified_10)` \n
        `PIL_save(simplified_5, "simplified_5")` \n
    """
    return SimplifyColorV4(colorcount, input)



def chain_center(chain: list):
    """ ### Returns the center of a chain of colors"""
    return (int(np.mean([rgb[0] for rgb in chain])),
            int(np.mean([rgb[1] for rgb in chain])),
            int(np.mean([rgb[2] for rgb in chain])))

def cluster_centers(colorset: set):
    """ ### Returns the centers of the clusters of colors"""
    chains = []
    subchain = set()
    for i in colorset:
        if not any(i in sublist for sublist in chains):
            chaincheck = i
            while closest_color_strict(chaincheck, colorset) not in subchain and not any(closest_color_strict(chaincheck, colorset) in sublist2 for sublist2 in chains):
                chaincheck = closest_color_strict(chaincheck, colorset)
                subchain.add(chaincheck)
            subchain.add(i)
            chains.append(subchain)
            subchain = set()
    print('Simplified down to', len(chains), 'colors.')
    chain_centers = [chain_center(chain) for chain in chains]
    return chain_centers

def SimplifyColorV5(input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [SimplifyColor V5](https://www.github.com/EgeEken/Simplify-Color) algorithm to the given image
    This program recreates the input image using less colors, colors are chosen by the chaining algorithm
    The complex algorithm causes the program to be more accurate, but also significantly slower.
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV5(100, "image.png")` \n
        `simplified_50 = SimplifyColorV5(50, img)` \n
        `simplified_10 = SimplifyColorV5(10, matrix)` \n
        `simplified_5 = SimplifyColorV5(5, simplified_10)` \n
        `PIL_save(simplified_5, "simplified_5")` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    colorset = cluster_centers(colorset_create(input))
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            res_pixels[x, y] = closest_color(matrix[y, x], colorset)
    return res


def is_inside(x, y, matrix, dcount):
    """ ### Checks if the given pixel is covered by borders in the image, True if covered, False if not, used for fill object"""
    if matrix[y, x] == (0,0,0):
        return True
    f = int(round(dcount/8, 0))
    for xi in range(-f, 1 + f):
        for yi in range(-f, 1 + f):
            xcheck = x
            ycheck = y
            while True:
                if (xi == 0 and yi == 0) or (yi!= 0 and xi/yi == 1 and not (xi == 1 or xi  == -1)):
                    break
                xcheck += xi
                ycheck += yi
                try:
                    a = matrix[ycheck, xcheck] == (0,0,0)
                except IndexError:
                    #print('image edge', (xcheck, ycheck), 'reached on direction:', (xi, yi))
                    return False 
                if matrix[ycheck, xcheck] == (0,0,0):
                    #print('border', (xcheck, ycheck), 'found on direction:', (xi, yi))
                    break
                if xcheck < 0 or ycheck < 0:
                    #print('image edge', (xcheck, ycheck), 'reached on direction:', (xi, yi))
                    return False
    return True

def is_inside_simplified(x, y, simplifiedmatrix, dcount):
    """ ### Checks if the given pixel is covered by borders in the image, True if covered, False if not, used for fill object"""
    if simplifiedmatrix[y, x] == 1:
        return True
    f = int(round(dcount/8, 0))
    for xi in range(-f, 1 + f):
        for yi in range(-f, 1 + f):
            xcheck = x
            ycheck = y
            while True:
                if (xi == 0 and yi == 0) or (yi!= 0 and xi/yi == 1 and not (xi == 1 or xi  == -1)):
                    break
                xcheck += xi
                ycheck += yi
                try:
                    a = simplifiedmatrix[ycheck, xcheck] == 1
                except IndexError:
                    #print('image edge', (xcheck, ycheck), 'reached on direction:', (xi, yi))
                    return False 
                if simplifiedmatrix[ycheck, xcheck] == 1:
                    #print('border', (xcheck, ycheck), 'found on direction:', (xi, yi))
                    break
                if xcheck < 0 or ycheck < 0:
                    #print('image edge', (xcheck, ycheck), 'reached on direction:', (xi, yi))
                    return False
    return True

def Fill_Object(input: str | np.ndarray | Image.Image, dcount: int = 8) -> Image.Image:
    """ ### Applies the [Fill Object](https://www.github.com/EgeEken/Fill-Object) algorithm to the given image
    Takes a black and white image and a direction count as input, and fills hollow objects with black by checking in {dcount, 8 by default} directions around each pixel to see if they are covered.
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `filled = fill_object("image.png")` \n
        `filled_16d = fill_object(img, 16)` \n
        `filled_4d = fill_object(matrix, 4)` \n
        `save(filled_4d, "image_filled_4d")` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            if is_inside(x,y, matrix, dcount):
                res_pixels[x, y] = (0,0,0)
    return res


def is_grain(simplifiedmatrix, x, y):
    """ ### Checks if the given pixel is a grain, used for grain detection"""
    if simplifiedmatrix[y, x] == 0:
        return False
    neighbors = around(simplifiedmatrix, x, y)
    for c in neighbors:
        if c == 1:
            return False
    return True

def Detect_Object(input: str | np.ndarray | Image.Image, threshold: int, dcount: int = 8, autobackground: bool = True, backcolor: tuple = None, graindetection: bool = False) -> Image.Image:
    """ ### Applies the [Detect Object](https://www.github.com/EgeEken/Detect-Object) algorithm to the given image
    Takes an image, a threshold for the Simplify algorithm, a direction count for the Fill Object algorithm as input, a bool for automatic background color (true by default), a background color tuple (None by default) and a grain detection bool (false by default)
    
    Returns an image where the background has been cropped, replaced by the background color (either automatically found or given) and the object is in focus.
    
    Usage: \n
        `img = PIL_open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `detected = Detect_Object("image.png", 100)` \n
        `detected_16d = Detect_Object(img, 100, 16)` \n
        `detected_4d_red_bg = Detect_Object(matrix, 100, 4, False, (255, 0, 0))` \n
        `save(detected_4d, "image_detected_4d")` \n
    """
    if type(input) == str:
        matrix = matrix_create(PIL_open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    width = matrix.shape[1]
    height = matrix.shape[0]
    
    simplifiedmatrix = create_contrast_matrix(matrix)
    for x in range(width):
        for y in range(height):
            if simplifiedmatrix[y, x] >= threshold:
                simplifiedmatrix[y, x] = 1
            else:
                simplifiedmatrix[y, x] = 0
    
    resmatrix = np.zeros((height, width), dtype=np.uint8)
    if autobackground:
        backtemp = np.array([0, 0, 0], dtype=np.uint8)
        backcount = 0
        if graindetection:
            for x in range(width):
                for y in range(height):
                    if simplifiedmatrix[y, x] == 1 or (is_inside_simplified(x, y, simplifiedmatrix, dcount) and not is_grain(simplifiedmatrix, x, y)):
                        resmatrix[y, x] = 1
                    else:
                        resmatrix[y, x] = 0
                        backtemp += np.array([matrix[y, x][0], matrix[y, x][1], matrix[y, x][2]], dtype=np.uint8)
                        backcount += 1

        else:
            for x in range(width):
                for y in range(height):
                    if simplifiedmatrix[y, x] == 1 or is_inside_simplified(x, y, simplifiedmatrix, dcount):
                        resmatrix[y, x] = 1
                    else:
                        resmatrix[y, x] = 0
                        backtemp += np.array([matrix[y, x][0], matrix[y, x][1], matrix[y, x][2]], dtype=np.uint8)
                        backcount += 1
                
        backcolor = tuple(backtemp // backcount)
        
    else:
        if graindetection:
            for x in range(width):
                for y in range(height):
                    if simplifiedmatrix[y, x] == 1 or (is_inside_simplified(x, y, simplifiedmatrix, dcount) and not is_grain(simplifiedmatrix, x, y)):
                        resmatrix[y, x] = 1
                    else:
                        resmatrix[y, x] = 0

        else:
            for x in range(width):
                for y in range(height):
                    if simplifiedmatrix[y, x] == 1 or is_inside_simplified(x, y, simplifiedmatrix, dcount):
                        resmatrix[y, x] = 1
                    else:
                        resmatrix[y, x] = 0
    
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = backcolor)
    res_pixels = PIL_load(res)
    for x in range(width):
        for y in range(height):
            if resmatrix[y, x] == 1:
                res_pixels[x, y] = matrix[y, x]
            else:
                res_pixels[x, y] = backcolor
    return res



def ReadVideo(filename: str, extension: str = "mp4") -> list:
    """ ### Reads a video file and returns a list of frames"""
    if filename[-3:] == extension:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(filename + "." + extension)
    res = []
    while True:
        ret, frame = cap.read()
        if ret:
            res.append(frame)
        else:
            break
    return res

def WriteVideo(filename: str, frames: list, fps: int = 30, extension: str = 'mp4', format: str = "mp4v"):
    """ ### Writes a list of frames to a video file"""
    size = frames[0].shape[:2][::-1]
    if filename[-3:] == extension:
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*format), fps, size)
    else:
        out = cv2.VideoWriter(filename + "." + extension, cv2.VideoWriter_fourcc(*format), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def video_to_txt(frames: list, filename: str) -> np.ndarray:
    """ ### Turns the given frame into a black and white version, and saves the colors in a textfile where the resulting black and white pixels are written in one line"""
    if filename[-4:] != ".txt":
        filename += ".txt"
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    for i in range(len(frames)):
        with open(filename, "w") as f:
            for y in range(height):
                for x in range(width):
                    f.write(str(frames[i][y, x]) + ",")
            f.write(os.linesep)
                   
def BnW_Video(filename: str, extension: str = "mp4") -> list:
    """ ### Normalizes the colors of a video while reading it into the given number of black to white colors and returns a cv2 video list"""
    if filename[-3:] == extension:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(filename + "." + extension)
    res = []
    while True:
        ret, frame = cap.read()
        if ret:
            res.append(cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        else:
            break
    return res

def distance_cv2(c1: np.ndarray, c2: np.ndarray) -> float:
    """ ### Returns the distance between two colors in RGB space, takes 2 cv2 colors instead of two tuples. 
    Returns a float
    Usage: \n
        `distance_cv2(np.array([0, 0, 0]), np.array([255, 255, 255]))` \n
        `video = ReadVideo("video")` \n
        `distance_cv2(video[0][0, 0], video[20][0, 0])` \n
    """
    if c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2]:
        return 0
    return math.sqrt(((float(c2[0])-float(c1[0]))**2+(float(c2[1]) - float(c1[1]))**2+(float(c2[2]) - float(c1[2]))**2))

def check_contrast_cv2(matrix: np.ndarray, x: int, y: int) -> float:
    """ ### Returns the contrast value of the pixel at matrix[y, x]
    Calculated as the sum of the distance between the pixel and its neighbors
    """
    if type(matrix[0, 0]) == float:
        return matrix
    neighbors = around(matrix, x, y)
    res = 0.0
    for i in neighbors:
        res += distance_cv2(matrix[y, x], i)
    return res

def create_contrast_matrix_cv2(matrix: np.ndarray) -> np.ndarray:
    """ ### Returns the contrast matrix of the given matrix
    
    Usage: \n
        `img1 = cv2.imread("image.png")` \n	
        `img2 = ReadVideo("video.mp4")[0]` \n
        `contrast_matrix = create_contrast_matrix_cv2(img1)` \n   
        `contrast_matrix = create_contrast_matrix_cv2(img2)` \n
    """
    width = matrix.shape[0]
    height = matrix.shape[1]
    res = np.zeros((height,width), dtype = float)
    for x in range(width):
        for y in range(height):
            res[y, x] = check_contrast_cv2(matrix, y, x)
    return res
    
def Simplify_cv2(img: np.ndarray, threshold: float) -> np.ndarray:
    """ ### Simplifies the given cv2 image by the given threshold and returns a cv2 image"""
    width = img.shape[1]
    height = img.shape[0]
    res = np.zeros((width, height, 3), dtype = np.uint8)
    contrastmatrix = create_contrast_matrix_cv2(img)    
    for x in range(width):
        for y in range(height):
            if contrastmatrix[x, y] >= threshold:
                res[x, y] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                res[x, y] = np.array([255, 255, 255], dtype=np.uint8)
    return res

def SimplifyVideo(filename: str | list, threshold: float, extension: str = "mp4") -> list:
    """ ### Simplifies the given video file by the given threshold and returns a cv2 video list"""
    if type(filename) == str:
        if filename[-3:] == extension:
            cap = cv2.VideoCapture(filename)
        else:
            cap = cv2.VideoCapture(filename + "." + extension)
        res = []
        while True:
            ret, frame = cap.read()
            if ret:
                res.append(Simplify_cv2(frame, threshold))
            else:
                break
        return res
    elif type(filename) == list:
        res = []
        for i in range(len(filename)):
            res.append(Simplify_cv2(filename[i], threshold))
        return res
     
def DownsizeVideo(filename: str | list, newwidth: int, newheight: int, extension: str = "mp4") -> list:
    """ ### Downsizes the given video file to the given width and height and returns a cv2 video list"""
    if type(filename) == str:
        if filename[-3:] == extension:
            cap = cv2.VideoCapture(filename)
        else:
            cap = cv2.VideoCapture(filename + "." + extension)
        res = []
        while True:
            ret, frame = cap.read()
            if ret:
                res.append(cv2.resize(frame, (newwidth, newheight)))
            else:
                break
        return res
    elif type(filename) == list:
        res = []
        for i in range(len(filename)):
            res.append(cv2.resize(filename[i], (newwidth, newheight)))
        return res

def Downsize_cv2(img: np.ndarray, newwidth: int, newheight: int) -> np.ndarray:
    """ ### Downsizes the given cv2 image to the given width and height and returns a cv2 image"""
    return cv2.resize(img, (newwidth, newheight))

def BnW_cv2(img: np.ndarray) -> np.ndarray:
    """ ### Converts the given cv2 image to black and white and returns a cv2 image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def Create_Training_Matrix(filename: str, extension: str = "mp4") -> np.ndarray:
    """ ### Creates a training data matrix of the given video file for a neural network"""
    if filename[-3:] == extension:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(filename + "." + extension)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res = np.zeros((framecount, width * height), dtype = np.uint8)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            res[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(width * height)
            i += 1
        else:
            break
    return res

def Training_Matrix_cv2(img: np.ndarray) -> np.ndarray:
    """ ### Creates a training data matrix of the given cv2 image for a neural network"""
    width = img.shape[1]
    height = img.shape[0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(width * height)

def Training_Matrix_cv2_GRAY(img: np.ndarray) -> np.ndarray:
    """ ### Creates a training data matrix of the given cv2 image for a neural network"""
    return img.reshape(img.shape[1] * img.shape[0])

def Create_Downsized_Training_Matrix(filename: str, newwidth: int, newheight: int, extension: str = "mp4") -> np.ndarray:
    """ ### Creates a training data matrix of a downsized version of the given video file for a neural network"""
    if filename[-3:] == extension:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(filename + "." + extension)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res = np.zeros((framecount, newwidth * newheight), dtype = np.uint8)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            res[i] = cv2.cvtColor(cv2.resize(frame, (newwidth, newheight)), cv2.COLOR_BGR2GRAY).reshape(newwidth * newheight)
            i += 1
        else:
            break
    
    return res

def Create_Downsized_Training_Video(filename: str, newwidth: int, newheight: int, extension: str = "mp4") -> list:
    """ ### Create_Downsized_Training_Matrix but returns a cv2 video list to be written so that it can be played"""
    if filename[-3:] == extension:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(filename + "." + extension)
    res = []
    while True:
        ret, frame = cap.read()
        if ret:
            res.append(cv2.cvtColor(cv2.cvtColor(cv2.resize(frame, (newwidth, newheight)), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        else:
            break
    return res

def ReadScreen() -> np.ndarray:
    """ ### Returns the current screen as a cv2 image"""
    return np.array(ImageGrab.grab(bbox=None), dtype=np.uint8)

def ReadDownsizedScreen(newwidth: int, newheight: int) -> np.ndarray:
    """ ### Downsizes and returns the current screen as a cv2 image"""
    return cv2.resize(ReadScreen(), (newwidth, newheight))

def ReadBnWScreen() -> np.ndarray:
    """ ### Returns the current screen as a cv2 image in black and white"""
    return cv2.cvtColor(cv2.cvtColor(ReadScreen(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def ReadDownsizedBnWScreen(newwidth: int, newheight: int) -> np.ndarray:
    """ ### Downsizes and returns the current screen as a cv2 image in black and white"""
    return cv2.cvtColor(cv2.resize(ReadScreen(), (newwidth, newheight)), cv2.COLOR_BGR2GRAY)

def ScreenMatrix_mc() -> np.ndarray:
    """ ### Returns the current screen as a training matrix for my minecraft player neural network"""
    return Training_Matrix_cv2_GRAY(ReadDownsizedBnWScreen(640, 360))
