from collections import abc
import torch
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
import einops

#TODO: Make vectorization cleaner.
#Right now, it jumbles the rectangles with respect to the input triangles - since we assume we're going to integrate it all anyway.
#To fix this we need to have another dimension for the geomtry queries: a batch dimension.
#Otherwise, we gonna have to loop for every pixel in screenspace using a python for-loop.
#TODO: Make sure the mag filter is nearest! Noise texture recursion wont work otherwise...
    #Perhaps: Maybe that can be done by rounding the UV values to the nearest texel - so we'll still integrate on the quads, but if they all fall into the same bin they're going to be the exact same
    #NOTE: You have to round to different intervals for every texel level

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
    assert x.ndim == 1, 'x should be a vector'
    assert y.ndim == 1, 'y should be a vector'
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
    assert x.ndim == 1, 'x should be a vector'
    assert y.ndim == 1, 'y should be a vector'
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


class IntegralTexture:
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
        cum_tex=cum_tex.cumsum(h_dim).cumsum(w_dim)
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

        assert x0.ndim == 1, 'x0 should be a vector'
        assert y0.ndim == 1, 'y0 should be a vector'
        assert x1.ndim == 1, 'x1 should be a vector'
        assert y1.ndim == 1, 'y1 should be a vector'
        assert len(x0) == len(y0) == len(x1) == len(y1), "x0, y0, x1, y1 must all have the same length"

        #Expensive assertions:
        #If I make bugs elsewhere, I might just sort x0,x1 and sort y0,y1...
        #It's possible to get a negative integral with this and thus a negative area otherwise...
        assert (y0<=y1).all()
        assert (x0<=x1).all()
        #x0, x1 = torch.minimum(x0,x1), torch.maximum(x0,x1)
        #y0, y1 = torch.minimum(y0,y1), torch.maximum(y0,y1)

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
        
        #Duplicate height and width across vectors
        xw = x0*0+w
        yh = x0*0+h

        #Calculating area is trivial
        area = dy*dx

        #This is based on the following fact, extended to 2d:
        #    ∀ a,b ∈ ℝ : a - b = (a % 1) + (⌊a⌋ - ⌊b⌋) - (b % 1)
        #    I have drawings in my notes about this. Sorry future reader lol. Maybe one day I'll draw it in unicode for you.
        #If this is a bottleneck, we can inline down to calculate_subpixel_weights to eliminate duplicate calculations
        s=self.cumsum
        sum = (
            +s(x1r, y1r)     +s(xw, y1r)*dxq     -s(x0r, y1r)     \
            -s(x1r, yh )*dyq +s(xw, yh )*dxq*dyq -s(x0r, yh )*dyq \
            -s(x1r, y0r)     -s(xw, y0r)*dxq     +s(x0r, y0r)     
        )

        assert sum.shape ==(c,n,)
        assert area.shape==(  n,)

        return sum, area



def _sort_xy_by_(*xys,axis='y'):
    """
    Anonymous helper function for sort_xy_by_y and sort_xy_by_x
    See their docstrings
    """
    assert axis in {'x','y'}

    assert xys, "Must provide at least one point-list"
    assert not len(xys)%2, "Number of args must be even, because each x must have a y. See docstring."

    xs=xys[0::2]
    ys=xys[1::2]

    assert all(x.ndim==1 for x in xs)
    assert all(y.ndim==1 for y in ys)
    assert len(set(map(len,xys)))==1, "All point-lists must have the same number of points"

    k=len(xys)//2 #Number of point-lists
    n=len(xys[0]) #Number of points per point-list

    xs=torch.stack(xs)
    ys=torch.stack(ys)
    assert xs.shape==ys.shape
    assert xs.shape==ys.shape==(k,n)

    key = dict(x=xs, y=ys)[axis]
    i = torch.argsort(key, dim=0)
    
    xs_sorted = torch.gather(xs, 0, i)
    ys_sorted = torch.gather(ys, 0, i)

    return list(itertools.chain(*zip(xs_sorted, ys_sorted)))


def sort_xy_by_y(*xys):
    """
    Signature: sort_xy_by_y(x1,y1,x2,y2, ..., xk,yk)
    Given a set of 2d x,y coordinates, return a new x0,y0,x1,y1 such that y0<=y1 for all points
    Assumes they are all torch tensors

    EXAMPLE:
        x0 = torch.tensor([1, 2, 3])
        y0 = torch.tensor([3, 1, 2])
        x1 = torch.tensor([4, 5, 6])
        y1 = torch.tensor([6, 4, 1])
        sx0, sy0, sx1, sy1 = sort_xy_by_y(x0, y0, x1, y1)
        print(as_numpy_array([sx0, sy0, sx1, sy1]))
        # OUTPUT:
        #  [[1 2 6]
        #   [3 1 1]
        #   [4 5 3]
        #   [6 4 2]]

    Checked: THIS FUNCTION IS CORRECT (only checked the above example though)
    """

    return _sort_xy_by_(*xys, axis= 'y')

def sort_xy_by_x(*xys):
    """
    Same idea as sort_xy_by_y, except along the x axis. See its docstring.

    EXAMPLE:
        x0 = torch.tensor([1, 2, 3])
        y0 = torch.tensor([3, 1, 2])
        x1 = torch.tensor([0, 1, 6])
        y1 = torch.tensor([6, 4, 1])
        sx0, sy0, sx1, sy1 = sort_xy_by_x(x0, y0, x1, y1)
        print(as_numpy_array([sx0, sy0, sx1, sy1]))
        # OUTPUT:
        # [[0 1 3]
        #  [6 4 2]
        #  [1 2 6]
        #  [3 1 1]]

    Checked: THIS FUNCTION IS CORRECT (only checked the above example though)
    """
    return _sort_xy_by_(*xys, axis= 'x')

def demo_integral_texture():
    """
    EXAMPLE OUTPUT:
        FIRST TEST. Remember, image[:,y,x] not image[:,x,y]!
        tensor([214.4391, 220.3378, 223.2823])
        tensor([[214.4392],
                [220.3378],
                [223.2823]])
        SECOND TEST: Just written a bit differently this time...
        Testing when x0 and y0 are 0 (aka integrate from corner)
        ic| tex.cum_tex[0,y1,x1]: tensor(732.9133)
        ic| tex.tex[0,:y1,:x1].sum(): tensor(732.9133)
        ic| tex.cumsum(torch.tensor([x1]), torch.tensor([y1]))[0, 0]: tensor(732.9133)
        ic| tex.integral(
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([x1]),
                torch.tensor([y1]),
            )[0][0,0]: tensor(732.9133)
        Testing when x0 and y0 are 0 (aka integrating a rectangle)
        ic| tex.tex[0,y0:y1,x0:x1].sum(): tensor(267.9160)
        ic| tex.integral(
                torch.tensor([x0]),
                torch.tensor([y0]),
                torch.tensor([x1]),
                torch.tensor([y1]),
            )[0][0,0]: tensor(267.9159)
        Wrapping tests...
        These should increase by the same amount each...
        But when we shift both bounds by the same amount it shouldnt change...
        Right now I did the math manually. It checks out.
        ic| "Testing x bounds...": 'Testing x bounds...'
            tex_integral_test(33 - 200, 22, 55 - 200, 44): tensor(246.5096)
            tex_integral_test(33 - 100, 22, 55 - 100, 44): tensor(246.5096)
            tex_integral_test(33, 22, 55, 44): tensor(246.5096)
            tex_integral_test(33, 22, 55 + 100, 44): tensor(1349.7982)
            tex_integral_test(33, 22, 55 + 200, 44): tensor(2453.0874)
        ic| "Testing y bounds...": 'Testing y bounds...'
            tex_integral_test(22, 33 - 200, 44, 55 - 200): tensor(249.0453)
            tex_integral_test(22, 33 - 100, 44, 55 - 100): tensor(249.0453)
            tex_integral_test(22, 33, 44, 55): tensor(249.0453)
            tex_integral_test(22, 33, 44, 55 + 100): tensor(-3045.3459)
            tex_integral_test(22, 33, 44, 55 + 200): tensor(-6339.7373)
         >>> 2453.0874-1349.7982
        ans = 1103.2892
         >>> 1349.7982-246.5096
        ans = 1103.2885999999999
        #Good, they're the same...
    """



    print('FIRST TEST. Remember, image[:,y,x] not image[:,x,y]!')
    image=torch.rand(3,100,100)

    x0=torch.tensor(random_ints(10,0,100))
    x1=torch.tensor(random_ints(10,0,100))
    y0=torch.tensor(random_ints(10,0,100))
    y1=torch.tensor(random_ints(10,0,100))

    #Make sure they're all valid rectangles...
    x0,x1 = torch.minimum(x0,x1), torch.maximum(x0,x1)
    y0,y1 = torch.minimum(y0,y1), torch.maximum(y0,y1)

    i=random_index(10)
    print(
        image[
            :,
            y0[i] : y1[i],
            x0[i] : x1[i],
        ].sum((1,2))
    )


    tex=IntegralTexture(image)
    print(
        tex.integral(
            x0[i][None],
            y0[i][None],
            x1[i][None],
            y1[i][None],
        )[0]
    )

    print('SECOND TEST: Just written a bit differently this time...')
    from icecream import ic
    print("Testing when x0 and y0 are 0 (aka integrate from corner)")
    x1=33
    y1=44
    ic(tex.cum_tex[0,y1,x1])
    ic(tex.tex[0,:y1,:x1].sum())
    ic(tex.cumsum(torch.tensor([x1]), torch.tensor([y1]))[0, 0])
    ic(
        tex.integral(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([x1]),
            torch.tensor([y1]),
        )[0][0,0]
    )

    print("Testing when x0 and y0 are 0 (aka integrating a rectangle)")
    x0=10
    y0=20
    ic(tex.tex[0,y0:y1,x0:x1].sum())
    ic(
        tex.integral(
            torch.tensor([x0]),
            torch.tensor([y0]),
            torch.tensor([x1]),
            torch.tensor([y1]),
        )[0][0,0]
    )


    def tex_integral_test(x0, y0, x1, y1):
        return tex.integral(
            torch.tensor([x0]),
            torch.tensor([y0]),
            torch.tensor([x1]),
            torch.tensor([y1]),
        )[0][0, 0]

    print("Wrapping tests...")
    print("These should increase by the same amount each...")
    print("But when we shift both bounds by the same amount it shouldnt change...")
    print("Right now I did the math manually. It checks out.")
    ic(
        "Testing x bounds...",
        tex_integral_test(33 - 200, 22, 55 - 200, 44),
        tex_integral_test(33 - 100, 22, 55 - 100, 44),
        tex_integral_test(33, 22, 55, 44),
        tex_integral_test(33, 22, 55 + 100, 44),
        tex_integral_test(33, 22, 55 + 200, 44),
    )
    ic(
        "Testing y bounds...",
        tex_integral_test(22, 33 - 200, 44, 55 - 200),
        tex_integral_test(22, 33 - 100, 44, 55 - 100),
        tex_integral_test(22, 33, 44, 55),
        tex_integral_test(22, 33, 44, 55 + 100),
        tex_integral_test(22, 33, 44, 55 + 200),
    )



def htraps_to_inner_rects(yb, yt, xbl, xbr, xtl, xtr):
    """
    Gets the bounds of a rectangle inscribed entirely in a trapezoid who has two edges parallel to the x-axis (aka an htrap)
    Convention in this func: Lower y == bottom, higher y == top
    (Regardless of image space vs cartesian space etc)
    yb corresponds to y0 in other funcs
    yt corresponds to y1 in other funcs
    Returns y0, y1, x0, x1
    Checked: THIS FUNCTION IS CORRECT (checked via demo_htraps_to_inner_rects and demo_subdivide_htrap_rects visually)
    """

    assert yb .ndim==1, 'yb  should be a vector. Aka bot y.'
    assert yt .ndim==1, 'yt  should be a vector. Aka top y.'
    assert xbl.ndim==1, 'xbl should be a vector. Aka bot left x.'
    assert xbr.ndim==1, 'xbr should be a vector. Aka bot right x.'
    assert xtl.ndim==1, 'xtl should be a vector. Aka top left x.'
    assert xtr.ndim==1, 'xtr should be a vector. Aka top right x.'
    assert len(set(map(len,[yb, yt, xbl, xbr, xtl, xtr])))==1, "They should all have same length"
    n = len(yb)
    
    #Expensive assertions. Might replace yb/yt if my code is too buggy...
    assert (yb <=yt ).all()
    assert (xbl<=xbr).all()
    assert (xtl<=xtr).all()

    #o stands for output

    #Output bounds:
    y0=yb #The bot bound
    y1=yt #The top bound

    x0=torch.maximum(xtl, xbl)
    x1=torch.minimum(xtr, xbr)
    
    #Don't have negative area: x0 can never be larger than x1
    #This might happen if a given trapezoid is skewed enough
    #In this case, collapse x0 and x1 to their mean
    xμ=(x0+x1)/2
    x0=torch.minimum(x0, xμ)
    x1=torch.maximum(x1, xμ)
    
    assert (x0<=x1).all(), "Internal assertion that should never fail"
    assert (y0<=y1).all(), "Internal assertion that should never fail"
    assert x0.shape==x1.shape==y0.shape==y1.shape==(n,)

    return y0, y1, x0, x1

def demo_htraps_to_inner_rects():
    import matplotlib.pyplot as plt
    import torch


    # Generate random htrap bounds
    n = 1
    yb = torch.rand(n)
    yt = yb + torch.rand(n) * 0.5
    xbl = torch.rand(n)
    xbr = xbl + torch.rand(n) * 0.5
    xtl = torch.rand(n)
    xtr = xtl + torch.rand(n) * 0.5

    # Get the inscribed rectangle bounds
    y0, y1, x0, x1 = htraps_to_inner_rects(yb, yt, xbl, xbr, xtl, xtr)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the htrap
    ax.plot([xbl.item(), xbr.item()], [yb.item(), yb.item()], 'b-')  # Bottom edge
    ax.plot([xtl.item(), xtr.item()], [yt.item(), yt.item()], 'b-')  # Top edge
    ax.plot([xbl.item(), xtl.item()], [yb.item(), yt.item()], 'b-')  # Left edge
    ax.plot([xbr.item(), xtr.item()], [yb.item(), yt.item()], 'b-')  # Right edge

    # Plot the inscribed rectangle
    ax.plot([x0.item(), x1.item()], [y0.item(), y0.item()], 'r-')  # Bottom edge
    ax.plot([x0.item(), x1.item()], [y1.item(), y1.item()], 'r-')  # Top edge
    ax.plot([x0.item(), x0.item()], [y0.item(), y1.item()], 'r-')  # Left edge
    ax.plot([x1.item(), x1.item()], [y0.item(), y1.item()], 'r-')  # Right edge

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Show the plot
    plt.show()



def subdivide_htraps(w:int, yb, yt, xbl, xbr, xtl, xtr):
    """
    Takes in a set of htraps in the form of points, and returns sets of points
    w is the number of subdivisions. I chose this variable letter arbitrarily because w looks like two of something...
    Uses same naming conventions as htraps_to_inner_rects (see its docstring)
    If we need to optimize memory, we could avoid storing the duplicate coordinates here...
    I assume you'll be using these h-traps to generate rectangles and integrate, so I'll just keep the output flat...
    Checked: THIS FUNCTION IS CORRECT (checked via demo_subdivide_htrap_rects visually)
    """



    assert yb.ndim == xbl.ndim == xbr.ndim == yt.ndim == xtl.ndim == xtr.ndim == 1
    assert len(yb) == len(xbl) == len(xbr) == len(yt) == len(xtl) == len(xtr)
    n = len(yb)  # Get the number of htraps
    
    #Expensive assertions. Might replace yb/yt if my code is too buggy...
    assert (yb <=yt ).all()
    assert (xbl<=xbr).all()
    assert (xtl<=xtr).all()

    #For now I assume they're all the same device/dtype. Maybe I'll add an assertion sometime later.
    dtype  = yb.dtype
    device = yb.device

    alphas = torch.linspace(0, 1, w+1).to(device).to(dtype)
    alphas=alphas[:, None]
    betas  = 1-alphas

    ys  = (yb  * betas + yt  * alphas)
    xls = (xbl * betas + xtl * alphas)
    xrs = (xbr * betas + xtr * alphas)

    #o stands for output
    oyb =ys [ :-1].flatten(0,1)
    oxbl=xls[ :-1].flatten(0,1)
    oxbr=xrs[ :-1].flatten(0,1)
    oyt =ys [1:  ].flatten(0,1)
    oxtl=xls[1:  ].flatten(0,1)
    oxtr=xrs[1:  ].flatten(0,1)

    assert oyb.shape==oxbl.shape==oxbr.shape==oyt.shape==oxtl.shape==oxtr.shape==(w*n,)

    #Expensive internal assertions. 
    assert (oyb <=oyt ).all()
    assert (oxbl<=oxbr).all()
    assert (oxtl<=oxtr).all()

    return oyb, oyt, oxbl, oxbr, oxtl, oxtr



def demo_subdivide_htrap_rects():
    "A combined demo of both htraps_to_inner_rects and htraps_to_inner_rects. Cool visuals!"
    # Generate random htrap bounds
    n = 1
    yb = torch.rand(n)
    yt = yb + torch.rand(n) * 0.5
    xbl = torch.rand(n)
    xbr = xbl + torch.rand(n) * 0.5
    xtl = torch.rand(n)
    xtr = xtl + torch.rand(n) * 0.5

    # Randomly choose w between 1 and 20
    w = np.random.randint(1, 21)

    # Subdivide the htrap
    oyb, oyt, oxbl, oxbr, oxtl, oxtr = subdivide_htraps(w, yb, yt, xbl, xbr, xtl, xtr)

    # Get the inscribed rectangle bounds for each subdivided htrap
    y0, y1, x0, x1 = htraps_to_inner_rects(oyb, oyt, oxbl, oxbr, oxtl, oxtr)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the subdivided htraps
    for i in range(w):
        ax.plot([oxbl[i].item(), oxbr[i].item()], [oyb[i].item(), oyb[i].item()], 'b-')  # Bottom edge
        ax.plot([oxtl[i].item(), oxtr[i].item()], [oyt[i].item(), oyt[i].item()], 'b-')  # Top edge
        ax.plot([oxbl[i].item(), oxtl[i].item()], [oyb[i].item(), oyt[i].item()], 'b-')  # Left edge
        ax.plot([oxbr[i].item(), oxtr[i].item()], [oyb[i].item(), oyt[i].item()], 'b-')  # Right edge

    # Plot the inscribed rectangles for each subdivided htrap
    for i in range(w):
        ax.plot([x0[i].item(), x1[i].item()], [y0[i].item(), y0[i].item()], 'r-')  # Bottom edge
        ax.plot([x0[i].item(), x1[i].item()], [y1[i].item(), y1[i].item()], 'r-')  # Top edge
        ax.plot([x0[i].item(), x0[i].item()], [y0[i].item(), y1[i].item()], 'r-')  # Left e\codge
        ax.plot([x1[i].item(), x1[i].item()], [y0[i].item(), y1[i].item()], 'r-')  # Right edge

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Show the plot
    plt.show()


def ilerp(start, end, values):
    """
    Inverse linear interpolation between start and end values based on values.

    Returns weights such that torch.lerp(start, end, weights) == values

    Warning: Might have NaN's!

    Args:
        start (torch.Tensor): The starting values of the interpolation.
        end (torch.Tensor): The ending values of the interpolation.
        values (torch.Tensor): The interpolated values between start and end.

    Returns:
        torch.Tensor: The computed values that would yield the given interpolated values.
        
    EXAMPLE:
        start = torch.tensor([1.0, 2.0])
        end = torch.tensor([5.0, 10.0])
        interpolated_values = torch.tensor([3.0, 6.0])

        values = ilerp(start, end, interpolated_values)
        print(values)  # Output: tensor([0.5000, 0.5000])
    """
    return (values - start) / (end - start)

def tris_to_htraps(ax, ay, bx, by, cx, cy):
    """
    Given a triangle, returns two h-traps
    Like subdivide_htraps, the result will be totally flat because this is really just for integrals over textures anyway...
    No assumptions are made about the ordering of the points on these triangles! It is this function's responsibility to make sure they don't make negative areas etc.
    Checked: THIS FUNCTION IS CORRECT (checked via demo_triangles_to_htraps visually)
    """


    assert ax.ndim == ay.ndim == bx.ndim == by.ndim == cx.ndim == cy.ndim == 1
    assert len(ax) == len(ay) == len(bx) == len(by) == len(cx) == len(cy)
    n=len(ax)

    #Notation in this function: b, m, t stand for bottom middle and top - aka y0, y1, and y2 respectively.

    #Sort the points by height.
    xb, yb, xm, ym, xt, yt = sort_xy_by_y(ax, ay, bx, by, cx, cy)
    assert (yt>=ym).all(), "Internal assertion - sort_xy_by_y shouldn't fail"
    assert (ym>=yb).all(), "Internal assertion - sort_xy_by_y shouldn't fail"

    #The only case where we would get nan is when yb==ym==yt, so the alpha doesn't matter in that case. Might as well be 1/2 - why not.
    alpha = ilerp(yb, yt, ym).nan_to_num(.5)
    xm_other = torch.lerp(xb, xt, alpha)

    #Make sure the left and right points correctly identified.
    xml, xmr = torch.minimum(xm, xm_other), torch.maximum(xm, xm_other)

    #Now for the upper h-traps (a trapezoid with two idential points we consider a trapezoid here)
    Uyb  = ym
    Uyt  = yt
    Uxbl = xml
    Uxbr = xmr
    Uxtl = xt
    Uxtr = xt

    #And now the lower h-traps
    Lyb  = yb
    Lyt  = ym
    Lxbl = xb
    Lxbr = xb
    Lxtl = xml
    Lxtr = xmr

    #Now combine the lower and upper h-traps into one flat list of h-traps
    yb  = torch.cat((Lyb , Uyb ))
    yt  = torch.cat((Lyt , Uyt ))
    xbl = torch.cat((Lxbl, Uxbl))
    xbr = torch.cat((Lxbr, Uxbr))
    xtl = torch.cat((Lxtl, Uxtl))
    xtr = torch.cat((Lxtr, Uxtr))

    assert yb.shape==yt.shape==xbl.shape==xbr.shape==xtl.shape==xtr.shape==(n*2,)

    return yb, yt, xbl, xbr, xtl, xtr 

def demo_triangles_to_htraps():
    # Generate random triangle vertices
    n = 1
    ax = torch.rand(n)
    ay = torch.rand(n)
    bx = torch.rand(n)
    by = torch.rand(n)
    cx = torch.rand(n)
    cy = torch.rand(n)

    # Convert the triangles to htraps
    yb, yt, xbl, xbr, xtl, xtr = tris_to_htraps(ax, ay, bx, by, cx, cy)

    # Create a figure and axis
    fig, plt_ax = plt.subplots()

    # Plot the triangle
    plt_ax.plot([ax.item(), bx.item(), cx.item(), ax.item()], [ay.item(), by.item(), cy.item(), ay.item()], 'b-')

    # Plot the htraps
    for i in range(2):
        plt_ax.plot([xbl[i].item(), xbr[i].item()], [yb[i].item(), yb[i].item()], 'r-')  # Bottom edge
        plt_ax.plot([xtl[i].item(), xtr[i].item()], [yt[i].item(), yt[i].item()], 'r-')  # Top edge
        plt_ax.plot([xbl[i].item(), xtl[i].item()], [yb[i].item(), yt[i].item()], 'r-')  # Left edge
        plt_ax.plot([xbr[i].item(), xtr[i].item()], [yb[i].item(), yt[i].item()], 'r-')  # Right edge

    # Set equal aspect ratio
    plt_ax.set_aspect('equal')

    # Show the plot
    plt.show()

def demo_triangle_to_htraps_to_rects():
    # Generate random triangle vertices
    n = 1
    ax = torch.rand(n)
    ay = torch.rand(n)
    bx = torch.rand(n)
    by = torch.rand(n)
    cx = torch.rand(n)
    cy = torch.rand(n)

    # Convert the triangles to htraps
    yb, yt, xbl, xbr, xtl, xtr = tris_to_htraps(ax, ay, bx, by, cx, cy)

    # Randomly choose w between 1 and 20
    w = np.random.randint(1, 21)

    # Subdivide the htraps
    oyb, oyt, oxbl, oxbr, oxtl, oxtr = subdivide_htraps(w, yb, yt, xbl, xbr, xtl, xtr)

    # Get the inscribed rectangle bounds for each subdivided htrap
    y0, y1, x0, x1 = htraps_to_inner_rects(oyb, oyt, oxbl, oxbr, oxtl, oxtr)

    # Create a figure and axis
    fig, plt_ax = plt.subplots()

    # Plot the triangle
    plt_ax.plot([ax.item(), bx.item(), cx.item(), ax.item()], [ay.item(), by.item(), cy.item(), ay.item()], 'b-')

    # Plot the subdivided htraps
    for i in range(len(oyb)):
        plt_ax.plot([oxbl[i].item(), oxbr[i].item()], [oyb[i].item(), oyb[i].item()], 'r-')  # Bottom edge
        plt_ax.plot([oxtl[i].item(), oxtr[i].item()], [oyt[i].item(), oyt[i].item()], 'r-')  # Top edge
        plt_ax.plot([oxbl[i].item(), oxtl[i].item()], [oyb[i].item(), oyt[i].item()], 'r-')  # Left edge
        plt_ax.plot([oxbr[i].item(), oxtr[i].item()], [oyb[i].item(), oyt[i].item()], 'r-')  # Right edge

    # Plot the inscribed rectangles for each subdivided htrap
    for i in range(len(oyb)):
        plt_ax.plot([x0[i].item(), x1[i].item()], [y0[i].item(), y0[i].item()], 'g-')  # Bottom edge
        plt_ax.plot([x0[i].item(), x1[i].item()], [y1[i].item(), y1[i].item()], 'g-')  # Top edge
        plt_ax.plot([x0[i].item(), x0[i].item()], [y0[i].item(), y1[i].item()], 'g-')  # Left edge
        plt_ax.plot([x1[i].item(), x1[i].item()], [y0[i].item(), y1[i].item()], 'g-')  # Right edge

    # Set equal aspect ratio
    plt_ax.set_aspect('equal')

    # Show the plot
    plt.show()


def tris_to_rects(w, ax, ay, bx, by, cx, cy):
    """
    Given triangles, gives rectangles that approximate them both vertically and horizontally
    Checked: THIS FUNCTION IS CORRECT (checked via demo_tris_to_rects visually)
    """

    def helper(ax, ay, bx, by, cx, cy):
        yb, yt, xbl, xbr, xtl, xtr = tris_to_htraps(ax, ay, bx, by, cx, cy)
        oyb, oyt, oxbl, oxbr, oxtl, oxtr = subdivide_htraps(w, yb, yt, xbl, xbr, xtl, xtr)
        y0, y1, x0, x1 = htraps_to_inner_rects(oyb, oyt, oxbl, oxbr, oxtl, oxtr)
        return y0, y1, x0, x1

    Hy0, Hy1, Hx0, Hx1 = helper(ax, ay, bx, by, cx, cy) #From h-traps
    Vx0, Vx1, Vy0, Vy1 = helper(ay, ax, by, bx, cy, cx) #From v-traps: just use the h-trap algo and transpose x and y twice

    y0 = torch.cat((Hy0, Vy0))
    y1 = torch.cat((Hy1, Vy1))
    x0 = torch.cat((Hx0, Vx0))
    x1 = torch.cat((Hx1, Vx1))

    return y0, y1, x0, x1


def demo_tris_to_rects():
    # Generate random triangle vertices
    n = 1
    ax = torch.rand(n)
    ay = torch.rand(n)
    bx = torch.rand(n)
    by = torch.rand(n)
    cx = torch.rand(n)
    cy = torch.rand(n)

    # Randomly choose w between 1 and 20
    w = np.random.randint(1, 21)

    # Get the inscribed rectangle bounds for each subdivided htrap
    y0, y1, x0, x1 = tris_to_rects(w, ax, ay, bx, by, cx, cy)

    # Create a figure and axis
    fig, plt_ax = plt.subplots()

    # Plot the triangle
    plt_ax.plot([ax.item(), bx.item(), cx.item(), ax.item()], [ay.item(), by.item(), cy.item(), ay.item()], 'b-')

    # Plot the inscribed rectangles
    for i in range(len(y0)):
        plt_ax.plot([x0[i].item(), x1[i].item()], [y0[i].item(), y0[i].item()], 'g-')  # Bottom edge
        plt_ax.plot([x0[i].item(), x1[i].item()], [y1[i].item(), y1[i].item()], 'g-')  # Top edge
        plt_ax.plot([x0[i].item(), x0[i].item()], [y0[i].item(), y1[i].item()], 'g-')  # Left edge
        plt_ax.plot([x1[i].item(), x1[i].item()], [y0[i].item(), y1[i].item()], 'g-')  # Right edge

    # Set equal aspect ratio
    plt_ax.set_aspect('equal')

    # Show the plot
    plt.show()



def uv_mapping_demo():
    import rp

    uvl_image = rp.load_image('uv_maps/triton_uvl_demo.exr',use_cache=True)
    uvl_image = resize_image_to_fit(uvl_image,256,256,interp='nearest')
    texture_image = rp.load_image('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png',use_cache=True)
    texture_image = rp.as_float_image(rp.as_rgb_image(texture_image))

    device=select_torch_device(prefer_used=True)

    uvl_image     = rp.as_torch_image(uvl_image    ).to(device)
    texture_image = rp.as_torch_image(texture_image).to(device)
    C,TH,TW = texture_image.shape

    u=uvl_image[0,:,:]
    v=uvl_image[1,:,:]

    #We will use two triangles:
    #a  b
    #
    #c  d
    #
    #abc and bcd

    au=u[:-1,:-1]*TW
    av=v[:-1,:-1]*TH

    bu=u[:-1,1: ]*TW
    bv=v[:-1,1: ]*TH

    cu=u[1: ,:-1]*TW
    cv=v[1: ,:-1]*TH

    du=u[1: ,1: ]*TW
    dv=v[1: ,1: ]*TH

    OH,OW=au.shape

    output = torch.zeros(C,OH,OW).to(device)

    tex = IntegralTexture(texture_image)

    y,x = torch.meshgrid(torch.arange(OH),torch.arange(OW))

    x=x.flatten().to(device)
    y=y.flatten().to(device)

    #Linear texture filtering
    output[:,y,x] = query_image_at_points(
        texture_image,
        au[y,x],
        av[y,x],
    )
    
    

    return output





