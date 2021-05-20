try:
    from itertools import izip as zip
except ImportError:
    pass
from numpy import zeros,array,arange,ones,compress,diff,empty,fliplr,tile,sign,abs
from matplotlib.pyplot import colorbar,boxplot,plot,get_cmap,fill_between,contourf,figure
try:
   from pyproj import Geod
   pyprojFlag=True
except:
   logging.warning("Could not import pyproj")
   pyprojFlag=False
from matplotlib.colors import ColorConverter,LinearSegmentedColormap,to_rgb,ListedColormap
from scipy.stats.mstats import mquantiles
from scipy.special import erf
from operator import itemgetter
from matplotlib.ticker import MaxNLocator,AutoLocator
import logging

class surface_zoom:
    """Class with depth transformation function for zooming towards the
    ocean surface and its inverse in order to provide tick lables.
    Assumes negative z values."""

    def __init__(self,n=3):
        """Defines zoom level via exponent of the inverse power mappings of
        vertical levels of the form:
        (z**1/n), where n can be chosen by the user.

        Args:
            n(integer): exponent of mapping function
        """

        self._n=n

    def __func__(self,z):
        """Mapping function (inverse power function) applied to project actual
         depth.

        Args:
            z(float array): original depth coordinates

        Returns:
            projected depth coordinates (float array)
        """

        return sign(z)*abs(z)**(1./float(self._n))

    def inv(self,z):
        """
        Inverse mapping function to obtain original depth coordinates from
        projected ones.

        Args:
            z (float array): levels in projected coordinates

        Returns:
            Levels in original coordinates (float array).
        """

        return sign(z)*abs(z)**(float(self._n))

    def __call__(self,z):
        """Transformation of array of depth levels applying mapping function
        __func__.

        Args:
            z(array of floats): original depth values.

        Returns:
            mapping function applied on input z.
        """

        return self.__func__(z)

def discretizeColormap(colmap,N):
   """Constructs colormap with N discrete color levels from continous map.

   Args:
      colmap(matplotlib.colors.colormap or derived instance): colormap from which
        to pick discrete colours;
      N (integer): number of colour levels

   Returns:
     discrete colormap (matplotlib.colors.LinearSegmentedColormap).
   """
   cmaplist = [colmap(i) for i in range(colmap.N)]
   return colmap.from_list('Custom discrete cmap', cmaplist, N)

class hovmoeller:
  """Class for plotting of hovmoeller diagrams using the contourf function,
  using surface zoom (optionally).

  Attributes:
    zoom (surface_zoom instance): projection to use for vertical coordinate
    contours (matplotlib.contour.QuadContourSet): contour set with filled
        contour levels of plot
    contourlines (matplotlib.contour.QuadContourSet) contour set with contour
        lines
    ax (matplotlib.axes.Axes): Axes to be used for plot
  """

  def __init__(self,t,dz,Var,contours=10,ztype="z",orientation="up",surface_zoom=True,
        zoom_obj=surface_zoom(),ax=0,lineopts={},**opts):
    """Defines basic settings and geometry and plots a hovmoeller diagram.

    Args:
        t (integer, float or datetime 1D-array): horizontal coordinate.
        dz (float 1D-array): thickness of vertical coordinate levels or vertical
            coordinate depending on ztype argument.
        Var (integer or float 2D-array): data array with dimensions
            len(dz),len(t)
        contours (any object accepted as third argument by the contourf function):
            contour argument to pass to contourf function
        ztype (string): definition of dz (vertical coordinate) type. For ztype=-"dz"
            dz is interpreted as vertical cell thickness, otherwise as cell centres.
        orientation (string): if not "up", the vertical coordinate is flipped.
        surface_zoom (boolean): if True the vertical coordinate is projected
            using zoom_obj.
        ax (matplotlib.axes.Axes): Axes to be used for plot (if 0,
            creates a new figure and axes).
        lineopts (dictionary): dictionary with options for contour lines passed
            to the contourf function.
        **opts: keyword options passed to the contourf function.
    """
    self.zoom=zoom_obj
    if ztype=="dz": #z-variable gives zell thickness, else cell centre
      if dz.ndim==1:
        dz=tile(dz,[t.shape[0],1])
      z=-dz.cumsum(1)+.5*dz
    else:
      z=dz
      if dz.ndim==1:
        z=tile(z,[t.shape[0],1])
    if surface_zoom:
        z_orig=z
        z=self.zoom(z)
        logging.info("{}, {}".format(z[0],z_orig[0]))
    if orientation is not "up":
        Var=fliplr(Var)
    if t.ndim==1:
        t=tile(t,[z.shape[1],1]).T
    if not ax:
        ax=figure().add_subplot(111)
    self.contours=ax.contourf(t.T,z.T,Var.T,contours,**opts)
    if lineopts:
        self.contourlines=ax.contour(t.T,z.T,Var.T,contours,**lineopts)
    ax.yaxis.set_major_locator(AutoLocator())
    #ticks=ax.yaxis.get_major_locator().tick_values(z_orig.min(),z_orig.max())
    #ax.yaxis.set_ticks(zoom_obj(ticks))
    ticks=ax.get_yticks()
    if surface_zoom: ax.yaxis.set_ticklabels(["{}".format(self.zoom.inv(t)) for t in ticks])
    self.ax=ax

  def set_ticks(self,ticks,ticklables=()):
    """
    Sets ticks and ticklabels of vertical axis in hovmoeller diagram.

    Args:
        ticks (sequence of floats): positions of vertical ticks
        ticklables (sequence of strings): strings to be used as ticklables,
            if empty, these will be generated automatically from ticks.
    """
    self.ax.yaxis.set_ticks(self.zoom(ticks))
    if ticklables:
        self.ax.yaxis.set_ticklabels(ticklables)
    else:
        self.ax.yaxis.set_ticklabels("{}".format(t) for t in ticks)


def cmap_map(function,cmap):
    """ Manipulates a colormap by applying function on colormap cmap.
    This routine will break any discontinuous points in a colormap.

    Args:
        function: function to apply on cmap. Has to take a single argument with
            sequence of shape N,3 [r, g, b].
        cmap: colormap to apply function on.

    Returns:
        Linear Segmented Colormap with function applied.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red','green','blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = reduce(lambda x, y: x+y, step_dict.values())
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : array(cmap(step)[0:3])
    old_LUT = array(map( reduced_cmap, step_list))
    new_LUT = array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector
    return LinearSegmentedColormap('colormap',cdict,1024)

def discreteColors(noc,cols=['r','b','#FFF000','g','m','c','#FF8000','#400000','#004040','w','b']):
    """Generate a colormap from list of discrete input colours.

    Args:
        noc (integer): number of desired discrete colours.
        cols (list of matplotlib colours): sequence of colours to use. If
            < noc repeated up to required length.

    Returns:
        LinearSegmentedColormap with discrete colours.
    """
    while noc>len(cols): cols.extend(cols)
    cc=ColorConverter()
    clrs=[cc.to_rgba(col) for col in cols]
    ds=1./float(noc)
    splits=arange(0,1-.1*ds,ds)
    cdict={}
    cdentry=[(0.,clrs[0][0],clrs[0][0],)]
    for slev in arange(1,splits.shape[0]):
        cdentry.append((splits[slev],clrs[slev-1][0],clrs[slev][0]))
    cdentry.append((1.,clrs[slev][0],clrs[slev][0]))
    cdict['red']=tuple(cdentry)
    cdentry=[(0.,clrs[0][1],clrs[0][1],)]
    for slev in arange(1,splits.shape[0]):
	    cdentry.append((splits[slev],clrs[slev-1][1],clrs[slev][1]))
    cdentry.append((1.,clrs[slev][1],clrs[slev][1]))
    cdict['green']=tuple(cdentry)
    cdentry=[(0.,clrs[0][2],clrs[0][2],)]
    for slev in arange(1,splits.shape[0]):
        cdentry.append((splits[slev],clrs[slev-1][2],clrs[slev][2]))
    cdentry.append((1.,clrs[slev][2],clrs[slev][2]))
    cdict['blue']=tuple(cdentry)
    return LinearSegmentedColormap('discreteMap',cdict)

def discreteGreys(nog):
    """Generate a colormap with discrete number shades of grey.

    Args:
        nog (integer): number of levels of grey.

    Returns:
        LinearSegmentedColormap with discrete grey levels.
    """
    dg=1./(nog-1)
    greys=[str(g) for g in arange(0,1+dg*.1,dg)]
    return discreteColors(nog,greys)

def chlMapFun(Nlev=256):
    """Natural colour like colormap for chlorophyll-a plots.

    Args:
        Nlev (integer): number of colour levels

    Returns:
        LinearSegmentedColormap
    """

    return LinearSegmentedColormap('chlMap',
    {'blue':((0.,.1,.1),(.66,.95,.95),(1.,1.,1.)),
    'green':((0.,0.,0.),(.1,.0,.0),(.66,.95,.95),(1.,1.,1.)),
    'red':((0.,0.,0.),(.4,.0,.0),(1.,.9,.9))}, N=Nlev)

yearMap=LinearSegmentedColormap('yearMap',
    {'blue':((0.,.5,.5),(.125,1.,1.),(.375,0.,0.),(.625,0.,0.),(.875,0.,0.),(1.,.5,.5)),
    'green':((0.,.45,.45),(.125,0.,0.),(.375,1.,1.),(.625,0.,0.),(.875,1.,1.),(1.,.45,.45)),
    'red':((0.,.55,.55),(.125,0.,0.),(.375,0.,0.),(.625,1.,1.),(.875,1.,1.),(1.,.55,.55))}, N=256)

colorWheel=LinearSegmentedColormap('colorWheel',
    {'blue':((0.,1.,1.),(.25,0.,0.),(.5,0.,0.),(.75,0.,0.),(1.,1.,1.)),
    'green':((0.,0.,0.),(.25,1.,1.),(.5,1.,1.),(.75,0.,0.),(1.,0.,0.)),
    'red':((0.,0.,0.),(.25,0.,0.),(.5,1.,1.),(.75,1.,1.),(1.,0.,0.),)}, N=256)

zooMap2=LinearSegmentedColormap('zooMap2',
    {'blue':((0.,.1,.1),(.5,.25,.25),(1.,.9,.9)),
    'green':((0.,.0,.0),(.5,.25,.25),(1.,.9,.75)),
    'red':((0.,0.,0.),(1.,1.,1.))}, N=256)

zooMap=LinearSegmentedColormap('zooMap',
    {'blue':((0.,.1,.1),(.75,.4,.4),(.9,.46,.46),(1.,.75,.75)),
    'green':((0.,0.,0.),(.75,.375,.375),(.9,.45,.45),(1.,.75,.75)),
    'red':((0.,0.,0.),(.75,0.75,0.75),(1.,1.,1.))}, N=256)

chlMap2=LinearSegmentedColormap('chlMap2',
    {'blue':((0.,.1,.1),(.666,.7,.7),(1.,.1,.1)),
    'green':((0.,0.,0.),(.666,.7,.7),(1.,.6,.6)),
    'red':((0.,0.,0.),(.666,0.,0.),(1.,0.,0.))}, N=256)

SMap=LinearSegmentedColormap('SMap',
    {'blue':((0.,1.,1.),(1.,.2,.2)),
    'green':((0.,.875,.875),(1.,0.,0.)),
    'red':((0.,.5,.5),(1.,0.,0.))}, N=256)

SMap2=LinearSegmentedColormap('SMap',
    {'blue':((0.,1.,1.),(1.,.2,.2)),
    'green':((0.,.95,.95),(1.,0.,0.)),
    'red':((0.,.8,.8),(1.,0.,0.))}, N=256)

TMap=LinearSegmentedColormap('TMap',
    {'blue':((0.,.25,.25),(.5,.15625,.15625),(1.,.25,.25)),
    'green':((0.,0.,0.),(.5,0.,0.),(1.,1.,1.)),
    'red':((0.,0.,0.),(.5,.625,.625),(1.,1.,1.))}, N=256)

spmMap=LinearSegmentedColormap('spmMap',
    {'blue':((0.,.1,.1),(.75,.4,.4),(.9,.46,.46),(1.,.5,.5)),
    'green':((0.,0.,0.),(.75,.525,.525),(.9,.63,.63),(1.,.84,.84)),
    'red':((0.,0.,0.),(.75,0.5625,0.5625),(.9,.675,.675),(1.,.9,.9))}, N=256)

pmDarkMap=LinearSegmentedColormap('pmDarkMap',
    {'blue':((0.,.0,.0),(1.,.0,.0)),
    'green':((0.,0.,0.),(.495,0.,0.),(.5,0.,.0),(.505,.1,.1),(1.,1.,1.)),
    'red':((0.,1.,1.),(.495,.1,.1),(.5,0.,0.),(.505,0.,0.),(1.,.0,.0))}, N=256)

pmLightMap=LinearSegmentedColormap('pmLightMap',
    {'blue':((0.,.0,.0),(.495,.9,.9),(.5,1.,1.),(.505,.9,.9),(1.,.0,.0)),
    'green':((0.,.0,.0),(.495,.9,.9),(.5,1.,1.),(.505,1.,1.),(1.,1.,1.)),
    'red':((0.,1.,1.),(.495,1.,1.),(.5,1.,1.),(.505,.9,.9),(1.,.0,.0))}, N=256)

def asymmetric_divergent_cmap(point0,colorlow="xkcd:reddish",colorhigh="xkcd:petrol",color0_low="w",color0_up=0,n=256):
    """Construct LinearSegmentedColormap with linear gradient between end point colors and midcolors,
    where the mid-point may be moved to any relative position in between 0 and 1.

    Args:
        point0 (float): relative position between 0 and 1 where color0_low and color0_up apply
        colorlow (valid matplotlib color specification): color at low limit of colormap
        colorhigh (valid matplotlib color specification): color at high limit of colormap
        color0_low (valid matplotlib color specification): color at point0 approached from lower values
        color0_high (valid matplotlib color specification): color at point0 approached from higher value
        n (integer): color resolution

    Returns:
        LinearSegmentedColormap
    """
    rl,gl,bl=to_rgb(colorlow)
    r0,g0,b0=to_rgb(color0_low)
    rh,gh,bh=to_rgb(colorhigh)
    if color0_up:
        r0h,g0h,b0h=to_rgb(color0_up)
    else:
        r0h,g0h,b0h=to_rgb(color0_low)
    adcmap=LinearSegmentedColormap("DivAsCMap",
        {"red":((0.,rl,rl),(point0,r0,r0h),(1,rh,rh)),
        "green":((0.,gl,gl),(point0,g0,g0h),(1,gh,gh)),
        "blue":((0.,bl,bl),(point0,b0,b0h),(1,bh,bh))})
    return adcmap

def asymmetric_cmap_around_zero(vmin,vmax,**opts):
    """Construct LinearSegmentedColormap with linear gradient between end point colors and midcolors,
    where the mid-point is set at 0 in between vmin and vmax. Calls asymmetric_divergent_cmap
    to generate colormap.

    Args:
        vmin (float): minimum value for colormap
        vmax (float): maximum value for colormap
        opts: additional arguments passed to asymmetric_divergent_cmap

    Returns:
        Dictionary with cmap, vmin and vmax keys and respective definitions for unfolding into matplotlib
        color plot functions.
    """
    if vmin*vmax<0:
        p0=-vmin/(vmax-vmin)
    elif vmin+vmax<0:
        p0=1.
    else:
        p0=0.
    cm=asymmetric_divergent_cmap(p0,**opts)
    return {"cmap":cm,"vmin":vmin,"vmax":vmax}

if pyprojFlag:
   def getDistance(lon1,lat1,lon2,lat2,geoid='WGS84'):
      """Get distance betwwen two points on the earth surface.

      Args:
        lon1(float): longitude of first point
        lat1(float): latitude of first point
        lon2(float): longitude of second point
        lat2(float): latitude of second point
        geoid(string): geoid to use for projection

      Returns:
        distance in km (float)
      """
      g=Geod(ellps=geoid)
      return g.inv(lon1,lat1,lon2,lat2)[2]

def findXYDuplicates(x,y,d,preserveMask=False):
    """Finds duplicates in position for data given in x,y coordinates.

    Args:
       x (1d-array): X-coordinate
       y (1d-array): Y-coordinate
       d (1d-array): data defined on x,y
       preserveMask: flag to preserve mask of input data

    Returns:
       x,y,d and duplicate mask (0 where duplicate), sorted by x,y"""
    if not len(x)==len(y)==len(d):
      logging.warning("Longitude, Lattitude and data size don't match:")
      logging.warning("  Lon: "+str(len(x))+" Lat: "+str(len(y))+" Data: "+str(len(d)))
      return
    l=len(x)
    duplMask=ones(l)
    if preserveMask:
      xyd=[[xn,yn,n] for xn,yn,n in zip(x,y,arange(l))]
    else:
      xyd=[[xn,yn,dn] for xn,yn,dn in zip(x,y,d)]
    xyd.sort(key=itemgetter(0,1))
    elm1=xyd[0]
    for m,el in enumerate(xyd[1:]):
        if el[:2]==elm1[:2]:
            duplMask[m+1]=0
        elm1=el
    if preserveMask:
        sortList=[el[2] for el in xyd]
        xyd=array(xyd)[:,:2]
        x=xyd[:,0]
        y=xyd[:,1]
        d=take(d,sortList)
    else:
        xyd=array(xyd)
        x=xyd[:,0]
        y=xyd[:,1]
        d=xyd[:,2]
    return x,y,d,duplMask

def removeXYDuplicates(x,y,d,mask=False):
    """Removes duplicates in position for data given in x,y coordinates.

    Args:
       x (1d-array): X-coordinate
       y (1d-array): Y-coordinate
       d (1d-array): data defined on x,y
       mask: flag to preserve mask of input data

    Returns:
       x,y,d with duplicates removed, sorted by x,y"""
    x,y,d,Mask=findXYDuplicates(x,y,d,preserveMask=mask)
    if not all(Mask):
        d=compress(Mask,d)
        x=compress(Mask,x)
        y=compress(Mask,y)
        logging.info("Duplicate points removed in x/y map...")
    return x,y,d

def plotSmallDataRange(x,ycentre,yupper,ylower,linetype='-',color='r',fillcolor="0.8",edgecolor='k',alpha=1.):
    """Plots data range over x, given by series of centre, upper and lower values.

    Args:
        x (float,intger, datetime series): x coordinate
        ycentre (float,integer series): midlle or average values of range to show, plotted as line
        yupper (float,integer series): upper limit of values, plotted as upper edge line
        ylower (float,integer series): lower limit of values, plotted as lower edge line
        linetype (plot [fmt] argument): line format used for ycentre
        color (matplotlib color): colour used for ycentre line
        fillcolor (matplotlib color): colour used to fille space between yupper and ylower
        edgecolor (matplotlib color): colour used for limiting lines
        alpha (float): transparency level of filling colour
    """
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,linetype,color=color)

def plotDataRange(x,ycentre,yupper,ylower,yup,ylow,linetype='-',color='r',fillcolor="0.8",edgecolor='k',alpha=1.):
    """Plots data range over x, given by series of centre, upper and lower values.

    Args:
        x (float,intger, datetime series): x coordinate
        ycentre (float,integer series): midlle or average values of range to show, plotted as line
        yupper (float,integer series): upper limit of values, plotted as upper edge line
        ylower (float,integer series): lower limit of values, plotted as lower edge line
        yup (float,integer series): values on the higher end of range to show, plotted as dotted line
        ylow (float,integer series): values on the lower end of range to show, plotted as dotted line
        linetype (plot [fmt] argument): line format used for ycentre
        color (matplotlib color): colour used for ycentre line
        fillcolor (matplotlib color): colour used to fille space between yupper and ylower
        edgecolor (matplotlib color): colour used for limiting lines
        alpha (float): transparency level of filling colour
    """
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,linetype,color=color)
    plot(x,yup,':',color=edgecolor)
    plot(x,ylow,':',color=edgecolor)
def plotFullDataRange(x,ycentre,yupper,ylower,yup,ylow,yu,yl,color='r',fillcolor="0.8",edgecolor='k',alpha=1.):
    """Plots data range over x, given by series of centre, upper and lower values.

    Args:
        x (float,intger, datetime series): x coordinate
        ycentre (float,integer series): midlle or average values of range to show, plotted as line
        yupper (float,integer series): upper limit of values, plotted as upper edge line
        ylower (float,integer series): lower limit of values, plotted as lower edge line
        yup (float,integer series): values on the higher end of range to show, plotted as dashedline
        ylow (float,integer series): values on the lower end of range to show, plotted as dashed line
        yu (float,integer series): additional set of values on the higher end of range to show,
            plotted as dotted line
        yl (float,integer series): additional set of values on the lower end of range to show,
            plotted as dotted line
        linetype (plot [fmt] argument): line format used for ycentre
        color (matplotlib color): colour used for ycentre line
        fillcolor (matplotlib color): colour used to fille space between yupper and ylower
        edgecolor (matplotlib color): colour used for limiting lines
        alpha (float): transparency level of filling colour
    """
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,color=color)
    plot(x,yup,'--',color=edgecolor)
    plot(x,ylow,'--',color=edgecolor)
    plot(x,yu,':',color=edgecolor)
    plot(x,yl,':',color=edgecolor)

def plotSpread(x,data,range=1,**opts):
    """Plot data spread over y along x coordinate.

    Args:
        x: coordinate of length N
        data: data of shape [N,K], spread is computed over K dimension, using quantiles.
        range: sets quantiles to use for plotting.
            1 - plot quantiles [.05,.25,.5,.75,.95]
            2 - plot quantiles [.01,.05,.25,.5,.75,.95,.99]
            else plot quantiles [.25,.5,.75,]
    """
    if range==2:
      probs=[.01,.05,.25,.5,.75,.95,.99]
    elif range==1:
      probs=[.05,.25,.5,.75,.95]
    else:
      probs=[.25,.5,.75,]
    mq=lambda d: mquantiles(d,probs)
    a=array([mq(d) for d in data]).transpose()
    if range==2:
        plotFullDataRange(x,a[3],a[2],a[4],a[1],a[5],a[0],a[6],**opts)
    elif range==1:
        plotDataRange(x,a[2],a[1],a[3],a[0],a[4],**opts)
    else:
        plotSmallDataRange(x,a[1],a[0],a[2],**opts)


def hcolorbar(shrink=0.5,pad=.05,**opts):
    """Horizontal colorbar.

    Args:
        shrink (float): shriking factor
        pad (float): padding to separate colorbar from other axes, expressed as
            fraction of original axes
        **opts: other options passed to colorbar function

    Returns:
        colorbar instance
    """
    return colorbar(orientation='horizontal',shrink=shrink,pad=pad,**opts)

moreColors={}
moreColors["fuchsia"]="#d63abc"
moreColors["rhubarb"]="#eb8381"
moreColors["blossom pink"]="#f5c4c7"
moreColors["apple red"]="#e13c32"
moreColors["brick red"]="#a02616"
moreColors["burgundy"]="#580f00"
moreColors["tangerine"]="#f08b1d"
moreColors["poppy"]="#fabe80"
moreColors["banana"]="#fff451"
moreColors["mari gold"]="#ffd202"
moreColors["chartreuse"]="#c2cd46"
moreColors["green apple"]="#60a557"
moreColors["olive green"]="#5f8225"
moreColors["spearmint"]="#70b3af"
moreColors["turquoise"]="#008873"
moreColors["teal"]="#005e6c"
moreColors["hunter green"]="#004027"
moreColors["midnight navy"]="#003661"
moreColors["lake blue"]="#0080b4"
moreColors["aqua blue"]="#009cc4"
moreColors["sky blue"]="#8ab1d4"
moreColors["lilac"]="#8477a6"
moreColors["beet"]="#731e71"
moreColors["grape"]="#47337a"
moreColors["cocoa"]="#483916"
moreColors["khaki"]="#c1bbad"
moreColors["cement"]="#79838b"
moreColors["jet black"]="#101a1d"

niceColorPalette=[moreColors[k] for k in ("midnight navy","mari gold","brick red","hunter green","tangerine","teal","beet","olive green","banana","lilac","cocoa","rhubarb","lake blue","poppy","burgundy","turquoise","apple red","chartreuse","sky blue","fuchsia","green apple","spearmint","grape","khaki","cement","blossom pink","jet black")]

_chl_rgb=[[0.03137254901960784, 0.11372549019607843, 0.34509803921568627],
    [0.03529411764705882, 0.11764705882352941, 0.35294117647058826],
    [0.0392156862745098, 0.11764705882352941, 0.3607843137254902],
    [0.043137254901960784, 0.12156862745098039, 0.3686274509803922],
    [0.047058823529411764, 0.12549019607843137, 0.3764705882352941],
    [0.050980392156862744, 0.12941176470588237, 0.3843137254901961],
    [0.054901960784313725, 0.12941176470588237, 0.39215686274509803],
    [0.058823529411764705, 0.13333333333333333, 0.4],
    [0.058823529411764705, 0.13725490196078433, 0.403921568627451],
    [0.06274509803921569, 0.1411764705882353, 0.4117647058823529],
    [0.06666666666666667, 0.1411764705882353, 0.4196078431372549],
    [0.07058823529411765, 0.1450980392156863, 0.42745098039215684],
    [0.07450980392156863, 0.14901960784313725, 0.43529411764705883],
    [0.0784313725490196, 0.15294117647058825, 0.44313725490196076],
    [0.08235294117647059, 0.15294117647058825, 0.45098039215686275],
    [0.08627450980392157, 0.1568627450980392, 0.4588235294117647],
    [0.09019607843137255, 0.1607843137254902, 0.4666666666666667],
    [0.09411764705882353, 0.16470588235294117, 0.4745098039215686],
    [0.09803921568627451, 0.16470588235294117, 0.4823529411764706],
    [0.10196078431372549, 0.16862745098039217, 0.49019607843137253],
    [0.10588235294117647, 0.17254901960784313, 0.4980392156862745],
    [0.10980392156862745, 0.17647058823529413, 0.5058823529411764],
    [0.11372549019607843, 0.17647058823529413, 0.5137254901960784],
    [0.11764705882352941, 0.1803921568627451, 0.5215686274509804],
    [0.11764705882352941, 0.1843137254901961, 0.5254901960784314],
    [0.12156862745098039, 0.18823529411764706, 0.5333333333333333],
    [0.12549019607843137, 0.18823529411764706, 0.5411764705882353],
    [0.12941176470588237, 0.19215686274509805, 0.5490196078431373],
    [0.13333333333333333, 0.19607843137254902, 0.5568627450980392],
    [0.13725490196078433, 0.2, 0.5647058823529412],
    [0.1411764705882353, 0.2, 0.5725490196078431],
    [0.1450980392156863, 0.20392156862745098, 0.5803921568627451],
    [0.1450980392156863, 0.20784313725490197, 0.5843137254901961],
    [0.1450980392156863, 0.21568627450980393, 0.5843137254901961],
    [0.1450980392156863, 0.2196078431372549, 0.5882352941176471],
    [0.1450980392156863, 0.2235294117647059, 0.592156862745098],
    [0.1450980392156863, 0.23137254901960785, 0.592156862745098],
    [0.1411764705882353, 0.23529411764705882, 0.596078431372549],
    [0.1411764705882353, 0.23921568627450981, 0.596078431372549],
    [0.1411764705882353, 0.24705882352941178, 0.6],
    [0.1411764705882353, 0.25098039215686274, 0.6039215686274509],
    [0.1411764705882353, 0.2549019607843137, 0.6039215686274509],
    [0.1411764705882353, 0.25882352941176473, 0.6078431372549019],
    [0.1411764705882353, 0.26666666666666666, 0.611764705882353],
    [0.1411764705882353, 0.27058823529411763, 0.611764705882353],
    [0.1411764705882353, 0.27450980392156865, 0.615686274509804],
    [0.1411764705882353, 0.2823529411764706, 0.615686274509804],
    [0.1411764705882353, 0.28627450980392155, 0.6196078431372549],
    [0.13725490196078433, 0.2901960784313726, 0.6235294117647059],
    [0.13725490196078433, 0.2980392156862745, 0.6235294117647059],
    [0.13725490196078433, 0.30196078431372547, 0.6274509803921569],
    [0.13725490196078433, 0.3058823529411765, 0.6313725490196078],
    [0.13725490196078433, 0.3137254901960784, 0.6313725490196078],
    [0.13725490196078433, 0.3176470588235294, 0.6352941176470588],
    [0.13725490196078433, 0.3215686274509804, 0.6352941176470588],
    [0.13725490196078433, 0.32941176470588235, 0.6392156862745098],
    [0.13725490196078433, 0.3333333333333333, 0.6431372549019608],
    [0.13725490196078433, 0.33725490196078434, 0.6431372549019608],
    [0.13333333333333333, 0.3411764705882353, 0.6470588235294118],
    [0.13333333333333333, 0.34901960784313724, 0.6509803921568628],
    [0.13333333333333333, 0.35294117647058826, 0.6509803921568628],
    [0.13333333333333333, 0.3568627450980392, 0.6549019607843137],
    [0.13333333333333333, 0.36470588235294116, 0.6549019607843137],
    [0.13333333333333333, 0.3686274509803922, 0.6588235294117647],
    [0.13333333333333333, 0.3764705882352941, 0.6627450980392157],
    [0.13333333333333333, 0.3803921568627451, 0.6666666666666666],
    [0.13333333333333333, 0.38823529411764707, 0.6666666666666666],
    [0.12941176470588237, 0.39215686274509803, 0.6705882352941176],
    [0.12941176470588237, 0.4, 0.6745098039215687],
    [0.12941176470588237, 0.40784313725490196, 0.6784313725490196],
    [0.12941176470588237, 0.4117647058823529, 0.6784313725490196],
    [0.12941176470588237, 0.4196078431372549, 0.6823529411764706],
    [0.12941176470588237, 0.4235294117647059, 0.6862745098039216],
    [0.12549019607843137, 0.43137254901960786, 0.6901960784313725],
    [0.12549019607843137, 0.4392156862745098, 0.6901960784313725],
    [0.12549019607843137, 0.44313725490196076, 0.6941176470588235],
    [0.12549019607843137, 0.45098039215686275, 0.6980392156862745],
    [0.12549019607843137, 0.4549019607843137, 0.7019607843137254],
    [0.12549019607843137, 0.4627450980392157, 0.7019607843137254],
    [0.12549019607843137, 0.47058823529411764, 0.7058823529411765],
    [0.12156862745098039, 0.4745098039215686, 0.7098039215686275],
    [0.12156862745098039, 0.4823529411764706, 0.7137254901960784],
    [0.12156862745098039, 0.48627450980392156, 0.7137254901960784],
    [0.12156862745098039, 0.49411764705882355, 0.7176470588235294],
    [0.12156862745098039, 0.4980392156862745, 0.7215686274509804],
    [0.12156862745098039, 0.5058823529411764, 0.7254901960784313],
    [0.11764705882352941, 0.5137254901960784, 0.7254901960784313],
    [0.11764705882352941, 0.5176470588235295, 0.7294117647058823],
    [0.11764705882352941, 0.5254901960784314, 0.7333333333333333],
    [0.11764705882352941, 0.5294117647058824, 0.7372549019607844],
    [0.11764705882352941, 0.5372549019607843, 0.7372549019607844],
    [0.11764705882352941, 0.5450980392156862, 0.7411764705882353],
    [0.11372549019607843, 0.5490196078431373, 0.7450980392156863],
    [0.11372549019607843, 0.5568627450980392, 0.7490196078431373],
    [0.11372549019607843, 0.5607843137254902, 0.7490196078431373],
    [0.11372549019607843, 0.5686274509803921, 0.7529411764705882],
    [0.11764705882352941, 0.5725490196078431, 0.7529411764705882],
    [0.12156862745098039, 0.5764705882352941, 0.7529411764705882],
    [0.12549019607843137, 0.5803921568627451, 0.7529411764705882],
    [0.13333333333333333, 0.5882352941176471, 0.7568627450980392],
    [0.13725490196078433, 0.592156862745098, 0.7568627450980392],
    [0.1411764705882353, 0.596078431372549, 0.7568627450980392],
    [0.1450980392156863, 0.6, 0.7568627450980392],
    [0.14901960784313725, 0.6039215686274509, 0.7568627450980392],
    [0.15294117647058825, 0.6078431372549019, 0.7568627450980392],
    [0.1568627450980392, 0.615686274509804, 0.7568627450980392],
    [0.1607843137254902, 0.6196078431372549, 0.7568627450980392],
    [0.16862745098039217, 0.6235294117647059, 0.7607843137254902],
    [0.17254901960784313, 0.6274509803921569, 0.7607843137254902],
    [0.17647058823529413, 0.6313725490196078, 0.7607843137254902],
    [0.1803921568627451, 0.6352941176470588, 0.7607843137254902],
    [0.1843137254901961, 0.6431372549019608, 0.7607843137254902],
    [0.18823529411764706, 0.6470588235294118, 0.7607843137254902],
    [0.19215686274509805, 0.6509803921568628, 0.7607843137254902],
    [0.19607843137254902, 0.6549019607843137, 0.7607843137254902],
    [0.20392156862745098, 0.6588235294117647, 0.7647058823529411],
    [0.20784313725490197, 0.6627450980392157, 0.7647058823529411],
    [0.21176470588235294, 0.6666666666666666, 0.7647058823529411],
    [0.21568627450980393, 0.6745098039215687, 0.7647058823529411],
    [0.2196078431372549, 0.6784313725490196, 0.7647058823529411],
    [0.2235294117647059, 0.6823529411764706, 0.7647058823529411],
    [0.22745098039215686, 0.6862745098039216, 0.7647058823529411],
    [0.23137254901960785, 0.6901960784313725, 0.7647058823529411],
    [0.23921568627450981, 0.6941176470588235, 0.7686274509803922],
    [0.24313725490196078, 0.7019607843137254, 0.7686274509803922],
    [0.24705882352941178, 0.7058823529411765, 0.7686274509803922],
    [0.25098039215686274, 0.7098039215686275, 0.7686274509803922],
    [0.2549019607843137, 0.7137254901960784, 0.7686274509803922],
    [0.2627450980392157, 0.7176470588235294, 0.7686274509803922],
    [0.27058823529411763, 0.7176470588235294, 0.7647058823529411],
    [0.2784313725490196, 0.7215686274509804, 0.7647058823529411],
    [0.28627450980392155, 0.7254901960784313, 0.7647058823529411],
    [0.29411764705882354, 0.7294117647058823, 0.7647058823529411],
    [0.30196078431372547, 0.7294117647058823, 0.7607843137254902],
    [0.30980392156862746, 0.7333333333333333, 0.7607843137254902],
    [0.3176470588235294, 0.7372549019607844, 0.7607843137254902],
    [0.3215686274509804, 0.7372549019607844, 0.7568627450980392],
    [0.32941176470588235, 0.7411764705882353, 0.7568627450980392],
    [0.33725490196078434, 0.7450980392156863, 0.7568627450980392],
    [0.34509803921568627, 0.7490196078431373, 0.7568627450980392],
    [0.35294117647058826, 0.7490196078431373, 0.7529411764705882],
    [0.3607843137254902, 0.7529411764705882, 0.7529411764705882],
    [0.3686274509803922, 0.7568627450980392, 0.7529411764705882],
    [0.3764705882352941, 0.7607843137254902, 0.7529411764705882],
    [0.3843137254901961, 0.7607843137254902, 0.7490196078431373],
    [0.39215686274509803, 0.7647058823529411, 0.7490196078431373],
    [0.4, 0.7686274509803922, 0.7490196078431373],
    [0.40784313725490196, 0.7686274509803922, 0.7450980392156863],
    [0.41568627450980394, 0.7725490196078432, 0.7450980392156863],
    [0.4235294117647059, 0.7764705882352941, 0.7450980392156863],
    [0.43137254901960786, 0.7803921568627451, 0.7450980392156863],
    [0.4392156862745098, 0.7803921568627451, 0.7411764705882353],
    [0.44313725490196076, 0.7843137254901961, 0.7411764705882353],
    [0.45098039215686275, 0.788235294117647, 0.7411764705882353],
    [0.4588235294117647, 0.788235294117647, 0.7372549019607844],
    [0.4666666666666667, 0.792156862745098, 0.7372549019607844],
    [0.4745098039215686, 0.796078431372549, 0.7372549019607844],
    [0.4823529411764706, 0.8, 0.7372549019607844],
    [0.49019607843137253, 0.8, 0.7333333333333333],
    [0.4980392156862745, 0.803921568627451, 0.7333333333333333],
    [0.5058823529411764, 0.807843137254902, 0.7333333333333333],
    [0.5176470588235295, 0.8117647058823529, 0.7333333333333333],
    [0.5254901960784314, 0.8156862745098039, 0.7294117647058823],
    [0.5333333333333333, 0.8196078431372549, 0.7294117647058823],
    [0.5411764705882353, 0.8196078431372549, 0.7294117647058823],
    [0.5529411764705883, 0.8235294117647058, 0.7294117647058823],
    [0.5607843137254902, 0.8274509803921568, 0.7254901960784313],
    [0.5686274509803921, 0.8313725490196079, 0.7254901960784313],
    [0.5764705882352941, 0.8352941176470589, 0.7254901960784313],
    [0.5882352941176471, 0.8392156862745098, 0.7254901960784313],
    [0.596078431372549, 0.8431372549019608, 0.7254901960784313],
    [0.6039215686274509, 0.8470588235294118, 0.7215686274509804],
    [0.611764705882353, 0.8470588235294118, 0.7215686274509804],
    [0.6235294117647059, 0.8509803921568627, 0.7215686274509804],
    [0.6313725490196078, 0.8549019607843137, 0.7215686274509804],
    [0.6392156862745098, 0.8588235294117647, 0.7215686274509804],
    [0.6470588235294118, 0.8627450980392157, 0.7176470588235294],
    [0.6588235294117647, 0.8666666666666667, 0.7176470588235294],
    [0.6666666666666666, 0.8705882352941177, 0.7176470588235294],
    [0.6745098039215687, 0.8745098039215686, 0.7176470588235294],
    [0.6823529411764706, 0.8745098039215686, 0.7137254901960784],
    [0.6941176470588235, 0.8784313725490196, 0.7137254901960784],
    [0.7019607843137254, 0.8823529411764706, 0.7137254901960784],
    [0.7098039215686275, 0.8862745098039215, 0.7137254901960784],
    [0.7176470588235294, 0.8901960784313725, 0.7137254901960784],
    [0.7294117647058823, 0.8941176470588236, 0.7098039215686275],
    [0.7372549019607844, 0.8980392156862745, 0.7098039215686275],
    [0.7450980392156863, 0.9019607843137255, 0.7098039215686275],
    [0.7529411764705882, 0.9019607843137255, 0.7098039215686275],
    [0.7647058823529411, 0.9058823529411765, 0.7058823529411765],
    [0.7725490196078432, 0.9098039215686274, 0.7058823529411765],
    [0.7803921568627451, 0.9137254901960784, 0.7058823529411765],
    [0.7843137254901961, 0.9137254901960784, 0.7058823529411765],
    [0.788235294117647, 0.9176470588235294, 0.7058823529411765],
    [0.796078431372549, 0.9176470588235294, 0.7058823529411765],
    [0.8, 0.9215686274509803, 0.7058823529411765],
    [0.803921568627451, 0.9215686274509803, 0.7058823529411765],
    [0.807843137254902, 0.9254901960784314, 0.7019607843137254],
    [0.8117647058823529, 0.9254901960784314, 0.7019607843137254],
    [0.8196078431372549, 0.9294117647058824, 0.7019607843137254],
    [0.8235294117647058, 0.9294117647058824, 0.7019607843137254],
    [0.8274509803921568, 0.9333333333333333, 0.7019607843137254],
    [0.8313725490196079, 0.9333333333333333, 0.7019607843137254],
    [0.8352941176470589, 0.9372549019607843, 0.7019607843137254],
    [0.8392156862745098, 0.9372549019607843, 0.7019607843137254],
    [0.8470588235294118, 0.9411764705882353, 0.7019607843137254],
    [0.8509803921568627, 0.9411764705882353, 0.7019607843137254],
    [0.8549019607843137, 0.9450980392156862, 0.7019607843137254],
    [0.8588235294117647, 0.9450980392156862, 0.6980392156862745],
    [0.8627450980392157, 0.9450980392156862, 0.6980392156862745],
    [0.8705882352941177, 0.9490196078431372, 0.6980392156862745],
    [0.8745098039215686, 0.9490196078431372, 0.6980392156862745],
    [0.8784313725490196, 0.9529411764705882, 0.6980392156862745],
    [0.8823529411764706, 0.9529411764705882, 0.6980392156862745],
    [0.8862745098039215, 0.9568627450980393, 0.6980392156862745],
    [0.8941176470588236, 0.9568627450980393, 0.6980392156862745],
    [0.8980392156862745, 0.9607843137254902, 0.6980392156862745],
    [0.9019607843137255, 0.9607843137254902, 0.6980392156862745],
    [0.9058823529411765, 0.9647058823529412, 0.6941176470588235],
    [0.9098039215686274, 0.9647058823529412, 0.6941176470588235],
    [0.9137254901960784, 0.9686274509803922, 0.6941176470588235],
    [0.9215686274509803, 0.9686274509803922, 0.6941176470588235],
    [0.9254901960784314, 0.9725490196078431, 0.6941176470588235],
    [0.9294117647058824, 0.9725490196078431, 0.6941176470588235],
    [0.9333333333333333, 0.9725490196078431, 0.6980392156862745],
    [0.9333333333333333, 0.9725490196078431, 0.7058823529411765],
    [0.9372549019607843, 0.9764705882352941, 0.7098039215686275],
    [0.9372549019607843, 0.9764705882352941, 0.7137254901960784],
    [0.9411764705882353, 0.9764705882352941, 0.7176470588235294],
    [0.9411764705882353, 0.9764705882352941, 0.7254901960784313],
    [0.9450980392156862, 0.9803921568627451, 0.7294117647058823],
    [0.9490196078431372, 0.9803921568627451, 0.7333333333333333],
    [0.9490196078431372, 0.9803921568627451, 0.7372549019607844],
    [0.9529411764705882, 0.9803921568627451, 0.7450980392156863],
    [0.9529411764705882, 0.9803921568627451, 0.7490196078431373],
    [0.9568627450980393, 0.984313725490196, 0.7529411764705882],
    [0.9568627450980393, 0.984313725490196, 0.7568627450980392],
    [0.9607843137254902, 0.984313725490196, 0.7647058823529411],
    [0.9607843137254902, 0.984313725490196, 0.7686274509803922],
    [0.9647058823529412, 0.9882352941176471, 0.7725490196078432],
    [0.9686274509803922, 0.9882352941176471, 0.7764705882352941],
    [0.9686274509803922, 0.9882352941176471, 0.7843137254901961],
    [0.9725490196078431, 0.9882352941176471, 0.788235294117647],
    [0.9725490196078431, 0.9882352941176471, 0.792156862745098],
    [0.9764705882352941, 0.9921568627450981, 0.796078431372549],
    [0.9764705882352941, 0.9921568627450981, 0.803921568627451],
    [0.9803921568627451, 0.9921568627450981, 0.807843137254902],
    [0.984313725490196, 0.9921568627450981, 0.8117647058823529],
    [0.984313725490196, 0.9921568627450981, 0.8156862745098039],
    [0.9882352941176471, 0.996078431372549, 0.8235294117647058],
    [0.9882352941176471, 0.996078431372549, 0.8274509803921568],
    [0.9921568627450981, 0.996078431372549, 0.8313725490196079],
    [0.9921568627450981, 0.996078431372549, 0.8352941176470589],
    [0.996078431372549, 1.0, 0.8431372549019608],
    [0.996078431372549, 1.0, 0.8470588235294118]]

NEO_chlorophyll_map = ListedColormap(_chl_rgb,name="NEO chlorophyll map")
