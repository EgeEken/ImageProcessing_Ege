import PIL
from PIL import Image
import time
import numpy as np
import imageio.v3 as iio3
import imageio.v2 as iio2
import math
import random
import cv2

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
    """ ### Opens an image file using PIL (Python Image Library)
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
    """ ### Loads the PIL.Image.Image object that you can create using the open() function, allowing you to iterate through and edit the pixels in it
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
    loaded = load(img_colors)
    width = img.size[0]
    height = img.size[1]
    res = np.zeros((height,width), dtype=tuple)
    for x in range(width):
        for y in range(height):
            res[y, x] = loaded[x,y]
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
    res = np.zeros((height,width))
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
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `brightened_50 = Brighten(0.5, "image.png")` \n
        `same_0 = Brighten(0, img)` \n
        `darkened_50 = Brighten(-0.5, matrix)` \n
    """
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

def Saturate_color(c: tuple, coef: float) -> tuple:
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
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `saturated_50 = Saturate(matrix, 0.5)` \n
        `desaturated_50 = Saturate(matrix, -0.5)` \n    
    """
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
            res[y, x] = Saturate_color(coef, matrix[y, x])
    return res

def BnW(input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Converts the image to black and white using the [Saturate](https://www.github.com/EgeEken/Saturate) function
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `black_and_white1 = BnW("image.png")` \n
        `black_and_white2 = BnW(matrix)` \n
        `black_and_white3 = BnW(img)` \n
    """
    return Saturate(-1, input)

def Normalize(input: str | np.ndarray | Image.Image) -> np.ndarray:
    """ ### Converts the image to black and white, returns a matrix of floats between 0 and 1, where 0 is black and 1 is white
    Primary intended use case is for neural networks
    
    Usage: \n
        `img = open("image.png")` \n
        `normalized = Normalize(img)` \n
        `print(normalized[0, 0]) #for the value of the top left pixel between 0 and 1` \n
    """
    bnw = BnW(input)
    res = np.zeros(bnw.shape, dtype=float)
    for x in range(bnw.shape[1]):
        for y in range(bnw.shape[0]):
            res[y, x] = bnw[y, x][0] / 255
    return res

def colorset_create(input: np.ndarray | Image.Image | str) -> set:
    """ ### Creates a set of all the different colors in an image matrix
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `colorset = colorset_create(matrix)` \n
        `print(len(colorset)) #for the number of colors` \n
    """
    if type(input) == str:
        matrix = matrix_create(open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    res = set()
    for i in matrix:
        res |= set(i)
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
        `img = open("image.png")` \n
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
    """ ### Returns a line of text representing the given image
    Primary intended use case is for neural networks"""
    if type(input) == str:
        matrix = matrix_create(open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    res = ""
    for i in matrix:
        for j in i:
            res += str(j)
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
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV4(100, "image.png")` \n
        `simplified_50 = SimplifyColorV4(50, img)` \n
        `simplified_10 = SimplifyColorV4(10, matrix)` \n
        `simplified_5 = SimplifyColorV4(5, simplified_10)` \n
        `save(simplified_5, "simplified_5")` \n
    """
    if type(input) == str:
        matrix = matrix_create(open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    if colorcount < 1:
        print("Warning: Color count is smaller than 1, setting it to 1")
        colorcount = 1
    colorset = np.random.choice(colorset_create(matrix), colorcount)
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    for x in range(width):
        for y in range(height):
            res[y, x] = closest_color(matrix[y, x], colorset)
    return res

def chain_center(chain: list):
    """ ### Returns the center of a chain of colors"""
    return (int(np.mean([chain[i][0] for i in range(len(chain))])),
            int(np.mean([chain[i][1] for i in range(len(chain))])),
            int(np.mean([chain[i][2] for i in range(len(chain))])))

def cluster_centers(colorset: set):
    """ ### Returns the centers of the clusters of colors"""
    chains = []
    subchain = set()
    for i in colorset:
        if not any(i in sublist for sublist in chains):
            chaincheck = i
            while closest_color_strict(colorset, chaincheck) not in subchain and not any(closest_color_strict(colorset, chaincheck) in sublist2 for sublist2 in chains):
                chaincheck = closest_color_strict(colorset, chaincheck)
                subchain.add(chaincheck)
            subchain.add(i)
            chains.append(subchain)
            subchain = set()
    print('Simplified down to', len(chains), 'colors.')
    chain_centers = [chain_center(chain) for chain in chains]
    return chain_centers

def Quantize(colorcount: int, input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [SimplifyColor V4](https://www.github.com/EgeEken/Simplify-Color) algorithm to the given image
    This program recreates the input image using the given number of colors, colors are sampled randomly from the image.
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV4(100, "image.png")` \n
        `simplified_50 = SimplifyColorV4(50, img)` \n
        `simplified_10 = SimplifyColorV4(10, matrix)` \n
        `simplified_5 = SimplifyColorV4(5, simplified_10)` \n
        `save(simplified_5, "simplified_5")` \n
    """
    return SimplifyColorV4(colorcount, input)

def SimplifyColorV5(input: str | np.ndarray | Image.Image) -> Image.Image:
    """ ### Applies the [SimplifyColor V5](https://www.github.com/EgeEken/Simplify-Color) algorithm to the given image
    This program recreates the input image using less colors, colors are chosen by the chaining algorithm
    The complex algorithm causes the program to be more accurate, but also significantly slower.
    
    Usage: \n
        `img = open("image.png")` \n
        `matrix = matrix_create(img)` \n
        `simplified_100 = SimplifyColorV5(100, "image.png")` \n
        `simplified_50 = SimplifyColorV5(50, img)` \n
        `simplified_10 = SimplifyColorV5(10, matrix)` \n
        `simplified_5 = SimplifyColorV5(5, simplified_10)` \n
        `save(simplified_5, "simplified_5")` \n
    """
    if type(input) == str:
        matrix = matrix_create(open(input))
    elif type(input) == Image.Image:
        matrix = matrix_create(input)
    elif type(input) == np.ndarray:
        matrix = input
    colorset = cluster_centers(colorset_create(input))
    width = matrix.shape[1]
    height = matrix.shape[0]
    res = PIL.Image.new(mode = "RGB", size = (width, height), color = (255, 255, 255))
    for x in range(width):
        for y in range(height):
            res[y, x] = closest_color(matrix[y, x], colorset)
    return res

def ReadVideo(filename: str) -> list:
    """ ### Reads a video file and returns a list of frames"""
    cap = cv2.VideoCapture(filename)
    res = []
    while True:
        ret, frame = cap.read()
        if ret:
            res.append(frame)
        else:
            break
    return res

def WriteVideo(filename: str, frames: list, fps: int, extension: str = 'mp4', format: str = "mp4v"):
    """ ### Writes a list of frames to a video file"""
    size = frames[0].shape[:2]
    if filename[-3:] == extension:
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*format), fps, size)
    else:
        out = cv2.VideoWriter(filename + "." + extension, cv2.VideoWriter_fourcc(*format), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def cv2_to_matrix(cv2_img: np.ndarray) -> np.ndarray:
    """ ### Converts a cv2 image to a matrix with tuples as elements instead of uint8 arrays"""
    res = np.zeros((cv2_img.shape[0], cv2_img.shape[1]), dtype = tuple)
    for i in range(cv2_img.shape[0]):
        for j in range(cv2_img.shape[1]):
            res[i, j] = tuple(cv2_img[i, j])
    return res

def cv2_video_to_matrix(cv2_video: list) -> np.ndarray:
    """ ### Converts a cv2 video to a matrix with tuples as elements instead of uint8 arrays"""
    res = np.zeros((len(cv2_video), cv2_video[0].shape[0], cv2_video[0].shape[1]), dtype = tuple)
    for i in range(len(cv2_video)):
        res[i] = cv2_to_matrix(cv2_video[i])
    return res

def matrix_to_cv2(matrix: np.ndarray) -> np.ndarray:
    """ ### Converts a matrix with tuples as elements to a cv2 image"""
    res = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype = np.uint8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(3):
                res[i, j, k] = matrix[i, j][k].astype(np.uint8)
    return res

def matrix_to_cv2_video(matrix_video: np.ndarray) -> list:
    """ ### Converts a matrix with tuples as elements to a cv2 video"""
    res = []
    for i in range(matrix_video.shape[0]):
        res.append(matrix_to_cv2(matrix_video[i]))
    return res

