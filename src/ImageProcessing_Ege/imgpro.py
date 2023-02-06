import PIL
from PIL import Image
import time
import numpy as np
import imageio.v3 as iio3
import imageio.v2 as iio2
import math

def save(img: Image.Image, filename: str, format: str = "PNG") -> None:
    """ ### Saves the PIL.Image.Image object as an image file (PNG by default) with the given name
    
    Usage: \n
        `img = open("image.png")` \n
        `var = load(img) ` \n
        `var[20, 10] = (255, 0, 0) `\n
        `save(img, "image_edited1")  ` \n
        `save(img, "image_edited2.jpg")` \n
        `save(img, "image_edited3", "JPG")` \n
        `This will change the pixel at (20, 10) to red and save the image as image_edited*.png ` \n
    
    You can check the [PIL documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html) for more info 
    """
    if filename.endswith(".png"):
        img.save(filename, "PNG")
    elif filename.endswith(".jpg"):
        img.save(filename, "JPG")
    else:
        img.save(filename + "." + format.lower(), format)

def open(filename: str) -> Image.Image:
    """ ### Opens an image file using PIL (Python Image Library) allowing you to iterate through the pixels in it
    Returns a PIL.Image.Image object
    
    Usage: \n
        `img1 = open("image.png")` \n
        `img2 = open("image")` \n
        
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
        
def load(image: Image.Image) -> Image.Image.load:
    """ ### Loads the PIL.Image.Image object that you can create using the open() function, allowing you to edit the pixels in it
    Returns a PIL.Image.Image.load object
    \n
    Usage: \n
        `img = open("image.png")` \n
        `var = load(img) ` \n
        `var[20, 10] = (255, 0, 0) `\n
        `save(img, "image_edited")  ` \n
        ` # This will change the pixel at (20, 10) to red and save the image as image_edited.png ` \n
        
    You can check the [PIL documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html) for more info 
    """
    return image.load()

def distance(c1: tuple, c2: tuple) -> float:
    """ ### Returns the distance between two colors in RGB space
    Returns a float
    Usage: \n
        `distance((30, 0, 0), (0, 40, 0)) == 50` \n
        `img = open("image.png")` \n
        `img_colors = img.convert("RGB") ` \n
        `distance(img_colors[0, 0], img_colors[1, 1])` \n
    """
    if c1 == c2:
        return 0
    return math.sqrt(((c2[0]-c1[0])**2+(c2[1] - c1[1])**2+(c2[2] - c1[2])**2))

def matrix_create(img: Image.Image) -> np.ndarray:
    """ ### Returns a numpy array of the image for faster processing
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
    """
    img_colors = img.convert('RGB')
    width = img.size[0]
    height = img.size[1]
    res = np.zeros(height,width)
    for x in range(width):
        for y in range(height):
            res[y, x] = img_colors[x,y]
    return res

def around(matrix: np.ndarray, x: int, y: int) -> list:
    """ ### Checks around matrix[y, x] and returns a list of the color values of the pixels around it
    Usage: \n
        `img = open("image.png")` \n
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
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `check_contrast(matrix, 10, 10)` \n
    """
    if type(matrix[0, 0]) == float:
        return matrix
    neighbors = around(matrix, x, y)
    res = 0
    for i in neighbors:
        res += distance(matrix[y, x], i)
    return res

def create_contrast_matrix(matrix: np.ndarray) -> np.ndarray:
    """ ### Returns a matrix of the contrast values of the pixels in the given matrix
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `contrast_matrix = create_contrast_matrix(matrix)` \n
    """
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = np.zeros(height, width)
    for x in range(width):
        for y in range(height):
            res[y, x] = check_contrast(matrix, x, y)
    return res

def Simplify(threshold: float, input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [Simplify](https://www.github.com/EgeEken/Simplify) algorithm to the given image)
    This results in a image with only black and white pixels, black pixels being the ones in high contrast points
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `contrast_matrix = create_contrast_matrix(matrix)` \n
        `simplified_400 = Simplify(400, "image.png")` \n
        `simplified_300 = Simplify(300, img)` \n
        `simplified_200 = Simplify(200, matrix)` \n	
        `simplified_100 = Simplify(100, contrast_matrix)` \n
    """
    contrastmatrix = None
    while contrastmatrix == None:
        if type(input) == str:
            contrastmatrix = create_contrast_matrix(matrix_create(open(input)))
        elif type(input) == Image.Image:
            contrastmatrix = create_contrast_matrix(matrix_create(input))
        elif type(input) == np.ndarray:
            contrastmatrix = create_contrast_matrix(input)
    width = contrastmatrix.shape[1]
    height = contrastmatrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    for x in range(width):
        for y in range(height):
            if contrastmatrix[y, x] > threshold:
                res[y, x] = (0, 0, 0)
    return res

def Brighten_color(coef: float, c: tuple) -> tuple:
    if coef == 0:
        return input
    elif coef > 1:
        coef = 1
    elif coef < -1:
        coef = -1
    r, g, b = c

    if coef < 0:
        return (int(r + r * coef), int(g + g * coef), int(b + b * coef))
    return (int(r + (255 - r) * coef ), int(g + (255 - g) * coef ), int(b + (255 - b) * coef ))

def Brighten(coef: float, input: str | np.ndarray) -> Image.Image:
    """### Brightens or darkens the image using the [Brighten](https://www.github.com/EgeEken/Brighten) function
    
    A coefficient of -1 would make the image completely black, while a coefficient of 1 would make the image completely white
    
    Usage: \n
        `img = open("image.png")` \n
        `brightened_50 = Brighten(0.5, "image.png")` \n
        `darkened_50 = Brighten(-0.5, "image.png")` \n
    """
    matrix = None
    while matrix == None:
        if type(input) == str:
            matrix = matrix_create(open(input))
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
    for x in range(width):
        for y in range(height):
            res[y, x] = Brighten_color(coef, matrix[y, x])
    return res
