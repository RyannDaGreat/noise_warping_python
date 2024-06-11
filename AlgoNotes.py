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
    
    Checked: THIS FUNCTION IS CORRECT (checked via demo_query_image_at_points)
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
    Image wraps around at the edges
    Inputs:
        image: CHW
        x: tensor of length N
        y: tensor of length N
    Returns:
        output: CN

    Checked: THIS FUNCTION IS CORRECT (checked via demo_query_image_at_points)
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

    Checked: THIS FUNCTION IS WORKS CORRECTLY (inspected visually)
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



def prepend_zeros(tensor, dim):
    """
    Prepends a single layer of zeros to the tensor along the specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): Dimension along which to prepend zeros.

    Returns:
        torch.Tensor: The tensor with a layer of zeros prepended along the specified dimension.
    """
    return torch.cat(
        (
            torch.zeros_like(tensor.select(dim, 0).unsqueeze(dim)),
            tensor
        ),
        dim=dim
    )


class CumulativeTexture:
    def __init__(self, tex):
        """
        Texture is given in CHW form
        """
        assert tex.ndim == 3, "tex must be in CHW format"

        #We are using CHW form
        h_dim = 1
        w_dim = 2

        
        self.tex=tex
        self.height=tex.shape[h_dim]
        self.width =tex.shape[w_dim]
        
        #Add zeros rows/columns to the texture
        cum_tex=tex
        cum_tex=prepend_zeros(cum_tex, h_dim)
        cum_tex=prepend_zeros(cum_tex, w_dim)
        cum_tex=tex.cumsum(h_dim).cumsum(w_dim)
        self.cum_tex=cum_tex

        #The cum_tex has one extra row and column, so its height and width each one more than that of tex
        assert self.cum_tex.shape[h_dim]==self.height+1
        assert self.cum_tex.shape[w_dim]==self.width +1

    def cumsum(self, x, y):
        """
        When x and y are ints, should be equivalent to self.tex[:x,:y].sum((1,2))
        When floating point, should interpolate
        Assumes x and y are in-bounds
        """

        #Expensive assertions:
        assert (x>=0         ).all()
        assert (y>=0         ).all()
        assert (x<=self.width).all()
        assert (y<=self.width).all()

        return query_image_at_points(self.cum_tex, x, y)

    def integral(self, x0, y0, x1, y1):
        """
        Integrates our given texture continuously in the given bounds
        Returns (sum, area)
        The bounds can be any real number - they wrap around the texture continuously
        """

        assert x0.ndim == 1, 'x0 should be a tensor'
        assert y0.ndim == 1, 'y0 should be a tensor'
        assert x1.ndim == 1, 'x1 should be a tensor'
        assert y1.ndim == 1, 'y1 should be a tensor'
        assert len(x0) == len(y0) == len(x1) == len(y1), "x0, y0, x1, y1 must all have the same length"

        #Expensive assertions:
        #If I make bugs elsewhere, I might just sort x0,x1 and sort y0,y1...
        #It's possible to get a negative integral with this and thus a negative area otherwise...
        assert (y0<=y1).all()
        assert (x0<=x1).all()

        c,h,w=self.tex.shape
        n=len(x0)

        #Remainders
        x0r=x0%w
        x1r=x1%w
        y0r=y0%h
        y1r=y1%h

        #Quotients
        x0q=x0//w
        x1q=x1//w
        y0q=y0//h
        y1q=y1//h

        #Deltas
        dx =x1 -x0
        dy =y1 -x0
        dxq=x1q-x0q
        dyq=y1q-y0q

        #Calculating area is trivial
        area = dy*dx

        #This is based on the following fact, extended to 2d:
        #    ∀ a,b ∈ ℝ : a - b = (a % 1) + (⌊a⌋ - ⌊b⌋) - (b % 1)
        #    I have drawings in my notes about this. Sorry future reader lol. Maybe one day I'll draw it in unicode for you.
        #If this is a bottleneck, we can inline down to calculate_subpixel_weights to eliminate duplicate calculations
        s=self.cumsum
        sum = (
            +s(x1r, y1r)     +s(w, y1r)*dxq     -s(x0r, y1r)     \
            -s(x1r, h  )*dyq +s(w, h  )*dxq*dyq -s(x0r, h  )*dyq \
            -s(x1r, y0r)     -s(w, y0r)*dxq     +s(x0r, y0r)     
        )

        assert sum.shape ==(n,c)
        assert area.shape==(n, )

        return sum, area




