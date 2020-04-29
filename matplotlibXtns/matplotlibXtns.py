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

#depthfun = lambda z: z**4/(z**3-2500**3) #assuming negative z
depthfun = lambda z: (1./z-1.)*z #assuming negative z
class surface_zoom:
    """Class with depth transformation function for zooming towards the
    ocean surface and its inverse in order to provide tick lables.
    Assumes negative z values."""

    def __init__(self,n=3):
        """Defines polyniomial mapping of vertical levels of the form:
        (z**1/n), where n can be chosen by the user.
        Args:
            n(intger): exponent of mapping function (default:3)
        """
        self._n=n

    __func__ = lambda self,z: sign(z)*abs(z)**(1./float(self._n))

    inv = lambda self,z: sign(z)*abs(z)**(float(self._n))

    def __call__(self,z):
        """Transformation of array of depth levels.
        Args:
            z(array of floats): original depth values."""
        return self.__func__(z)

def discretizeColormap(colmap,N):
   cmaplist = [colmap(i) for i in range(colmap.N)]
   return colmap.from_list('Custom discrete cmap', cmaplist, N)

class hovmoeller:
  def __init__(self,t,dz,Var,contours=10,ztype="z",orientation="up",surface_zoom=True,
        zoom_obj=surface_zoom(),ax=0,lineopts={},**opts):
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
    self.ax.yaxis.set_ticks(self.zoom(ticks))
    if ticklables:
        self.ax.yaxis.set_ticklabels(ticklables)
    else:
        self.ax.yaxis.set_ticklabels("{}".format(t) for t in ticks)


def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
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
    dg=1./(nog-1)
    greys=[str(g) for g in arange(0,1+dg*.1,dg)]
    return discreteColors(nog,greys)

def chlMapFun(Nlev=256):
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
        color0_high (valid matplotlib color specification): color at point0 approached from higher value,
            defaults to color0_low
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
      g=Geod(ellps=geoid)
      return g.inv(lon1,lat1,lon2,lat2)[2]

def findXYDuplicates(x,y,d,preserveMask=False):
    """Finds X,Y duplicates for data given in x,y coordinates.
       x: X-coordinate (1d-array)
       y: Y-coordinate (1d-array)
       d: data defined on x,y (1d-array)
    Returns sorted x,y,d and duplicate mask (0 where duplicate)"""
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
    """Removes X,Y duplicates for data given in x,y coordinates.
       x: X-coordinate (1d-array)
       y: Y-coordinate (1d-array)
       d: data defined on x,y (1d-array)
       Returns sorted x,y,d with duplicates removed"""
    x,y,d,Mask=findXYDuplicates(x,y,d,preserveMask=mask)
    if not all(Mask):
        d=compress(Mask,d)
        x=compress(Mask,x)
        y=compress(Mask,y)
        logging.info("Duplicate points removed in x/y map...")
    return x,y,d

def plotSmallDataRange(x,ycentre,yupper,ylower,linetype='-',color='r',fillcolor="0.8",edgecolor='k',alpha=1.,**args):
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,linetype,color=color)
def plotDataRange(x,ycentre,yupper,ylower,yup,ylow,linetype='-',color='r',fillcolor="0.8",edgecolor='k',alpha=1.,**args):
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,linetype,color=color)
    plot(x,yup,':',color=edgecolor)
    plot(x,ylow,':',color=edgecolor)
def plotFullDataRange(x,ycentre,yupper,ylower,yup,ylow,yu,yl,color='r',fillcolor="0.8",edgecolor='k',alpha=1.,**args):
    fill_between(x,yupper,ylower,color=fillcolor,edgecolor=edgecolor,alpha=alpha)
    plot(x,ycentre,color=color)
    plot(x,yup,'--',color=edgecolor)
    plot(x,ylow,'--',color=edgecolor)
    plot(x,yu,':',color=edgecolor)
    plot(x,yl,':',color=edgecolor)
def plotSpread(y,data,range=1,**opts):
    "plot data spread with y as variable dimension and x as sample dimension."
    if range==2:
      probs=[.01,.05,.25,.5,.75,.95,.99]
    elif range==1:
      probs=[.05,.25,.5,.75,.95]
    else:
      probs=[.25,.5,.75,]
    mq=lambda d: mquantiles(d,probs)
    a=array([mq(d) for d in data]).transpose()
    if range==2:
        plotFullDataRange(y,a[3],a[2],a[4],a[1],a[5],a[0],a[6],**opts)
    elif range==1:
        plotDataRange(y,a[2],a[1],a[3],a[0],a[4],**opts)
    else:
        plotSmallDataRange(y,a[1],a[0],a[2],**opts)



def hcolorbar(shrink=0.5,pad=.05,**opts):
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
