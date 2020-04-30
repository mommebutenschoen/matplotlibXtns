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
from matplotlib.colors import ColorConverter,LinearSegmentedColormap,to_rgb
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
  using surface zoom (optionally)."""

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
        logging.info(z[0],z_orig[0])
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
    ax.yaxis.set_ticklabels(["{}".format(self.zoom.inv(t)) for t in ticks])
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
            fraction of orgiinal axes 
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
