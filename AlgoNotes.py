import torch
import torch.nn.functional as F

def calculate_subpixel_weights(x, y):
    """
    Calculates subpixel weights for bilinear interpolation
    Inputs:
        x: tensor of length N
        y: tensor of length N
    Returns:
        fx, fy: floor
        cx, cy: ceil
        w1, w2, w3, w4: subpixel weights
    This function's math is explained here: 
        https://www.desmos.com/calculator/esool5qrrd
    """
    assert x.ndim == 1, 'x should be a tensor'
    assert y.ndim == 1, 'y should be a tensor'
    assert len(x) == len(y), "x and y must have the same length"

    fx = x.floor().long()
    fy = y.floor().long()
    cx = x.ceil ().long()
    cy = y.ceil ().long()

    rx = x - fx
    ry = y - fy
    qx = 1 - rx
    qy = 1 - ry

    w1 = qx * qy
    w2 = rx * qy
    w3 = qx * ry
    w4 = rx * ry

    return fx, fy, cx, cy, w1, w2, w3, w4


def query_image_at_points(image, x, y):
    """
    Queries points on an image using bilinear interpolation
    Inputs:
        image: CHW
        x: tensor of length N
        y: tensor of length N
    Returns:
        output: CN
    """
    assert image.ndim == 3, "image must be in CHW format"
    assert x.ndim == 1, 'x should be a tensor'
    assert y.ndim == 1, 'y should be a tensor'
    assert len(x) == len(y), "x and y must have the same length"

    c, h, w = image.shape
    n = len(x)

    # Calculate subpixel weights
    fx, fy, cx, cy, w1, w2, w3, w4 = calculate_subpixel_weights(x, y)

    # Wrap the image at the edges
    fx %= w
    cx %= w
    fy %= h
    cy %= h

    # Retrieve pixel values
    I1 = image[:, fy, fx]
    I2 = image[:, fy, cx]
    I3 = image[:, cy, fx]
    I4 = image[:, cy, cx]

    # Apply weights and sum
    output = w1 * I1 + w2 * I2 + w3 * I3 + w4 * I4

    assert output.shape == (c, n)

    return output



def demo_query_image_at_points():
    """
    Demo for query_image_at_points
    It upscales an image and shows a side-by-side
    Note how the pixels should wrap
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import requests
    from io import BytesIO
    import matplotlib.pyplot as plt
    import rp

    # Load the image from the URL
    url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    image = rp.load_image(url)
    image = rp.cv_resize_image(image, (64, 64))
    image = rp.as_float_image(image)
    torch_image = rp.as_torch_image(image)

    scale_factor = 5

    # Create a grid of points to query
    h, w = torch_image.shape[1:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(w * scale_factor), torch.arange(h * scale_factor)
    )
    x_grid = x_grid.float() / scale_factor
    y_grid = y_grid.float() / scale_factor

    # Query the image at the grid points
    upsampled_image = query_image_at_points(torch_image, x_grid.flatten(), y_grid.flatten())
    upsampled_image = upsampled_image.reshape(3, h * scale_factor, w * scale_factor)
    upsampled_image = rp.as_numpy_image(upsampled_image)

    # Display the original and upsampled images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(upsampled_image)
    ax2.set_title(f"Upsampled Image (Scale Factor: {scale_factor})")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()



class CumulativeTexture:
    def __init__(self, tex):
        """
        Texture is given in HWC form
        """
        self.tex=tex
        
        cum_tex=tex

        #Add a zeros row and column to the texture
        cum_tex=torch.concat((cum_tex[0][None]*0,cum_tex))
        cum_tex=torch.concat((cum_tex[:,0][:,None]*0,cum_tex))

        cum_tex=np.hstack(cum_tex[0]*0)
        cum_tex=tex.cumsum(0).cumsum(1)
        cum_tex=




def discrete_area_query(tex, x0, x1, y0, y1):
    #Tex is in HWC form - it is a matrix
    #y0, y1, x0, x1 are all int

    #How to query the value of a single pixel:
    if x1 is None:x1=x0+1
    if y1 is None:y1=y0+1

    assert (y0<=y1).all()
    assert (x0<=x1).all()

    height, width = tex.shape
    
    sum=0
    area=0

    #SUPPORT MODULAR ARITHMETIC

    #Make sure they're all non-negative
    #Drag them up so the bottom >=0
    y_shift = (y0%height)-y0
    x_shift = (x0%width )-x0
    y0 += y_shift
    y1 += y_shift
    x0 += x_shift
    x1 += x_shift
    assert (y0>=0).all()
    assert (y1>=0).all()
    assert (x0>=0).all()
    assert (x1>=0).all()
    assert (x0<width).all()
    assert (y0<width).all()


    #y1 might still be larger than x0...
    #CONCLUSION...just use webgl screenshots for now..........it's way faster than implementing this...

    tex_sum = tex.sum(0,1) #Might be cached
    tex_area = height*width 
    y_inner_wraps = (y1/height).floor() - (y0/height).ceil()
    x_inner_wraps = (x1/width ).floor() - (x0/width ).ceil()
    sum+=tex_sum*y_inner_wraps*x_inner_wraps
    area+=tex_area*y_inner_wraps*x_inner_wraps


    #THERE ARE SEVERAL PIECES YOU HAVE TO CONSIDER. DRAW IT OUT. THERE ARE g 
    y1-y_inner_wraps*height
    edge_y0_sum = 

    sum+=x_inner_wraps*edge_y0_sum

    corner_y0x0_sum
    sum+=corner_y0x0_sum
    area+=corner_y0x0_area



    



def discrete_area_query(tex, y0, y1, x0, x1):
    #Tex is in HW form - it is a matrix
    #y0, y1, x0, x1 are all int
    assert y0<=y1 and x0<=x1
    height, width = tex.shape
    
    sum=0
    area=0

    #SUPPORT MODULAR ARITHMETIC
    while y0<0:
        y0+=height
        y1+=height
    while x0<0:
        x0+=width
        x1+=width
    assert x0>=0 and y0>=0
    while y1>=height and x1>=width:
        sum+=tex[0:height,0:width].sum()
        area+=height*width
        y1-=height
        x1-=width
    assert y1<height or x1<width, 'Only one of the next two loops will trigger'
    while y1>=height:
        sum+=tex[0:height,x0:x1].sum()
        area+=height*(x1-x0)
        y1-=height
    while x1>=width:
        sum+=tex[y0:y1,0:width].sum()
        area+=(y1-y0)*width
        x1-=width
    
    #THE NORMAL CASE WHERE x0,x1,y0,y1 ARE INSIDE THE TEXTURE
    sum+=tex[y0:y1,x0:x1].sum()
    area+=(y1-y0)*(x1-x0)

    return sum, area
    


def continuous_area_query(tex, y0, y1, x0, x1):
    #y0, y1, x0, x1 are all float
    assert y0>=y1 and x0<=x1



    inner_x0 = x0.ceil()
    inner_x1 = x1.floor()
    inner_y0 = y0.ceil()
    inner_y1 = y1.floor()

    outer_x0 = x0.floor()
    outer_y0 = y0.ceil()
    outer_x1 = x1.ceil()
    outer_y1 = y1.floor()

