from __future__ import print_function
try:
    xrange(1)  # python2
except NameError:
    xrange = range  # python3
from netCDF4 import num2date as n4num2date
from netCDF4 import date2num as n4date2num
try:
    from itertools import izip as zip
except ImportError:
    pass
from numpy import zeros,ma,floor,ceil,where,array,arange,ones,compress,diff,empty,any,logical_not,fliplr,tile
from numpy.ma import masked_where,getmaskarray,getdata
from numpy.ma import array as marray
from scipy import linspace,meshgrid
from mpl_toolkits.basemap import Basemap,interp,maskoceans,shiftgrid,addcyclic
from datetime import datetime
from matplotlib.mlab import griddata
from matplotlib.dates import date2num,YearLocator,DateFormatter,MonthLocator
from matplotlib.pyplot import pcolormesh,axis,gca,text,colorbar,figure,boxplot,plot,get_cmap,fill_between,contourf
try:
 try:
  from netCDF4 import default_fillvals as fv
 except:
  from netCDF4 import _default_fillvals as fv
except:
  raise ImportError('Could not import netCDF4 default fill-values')
from time import time,clock
try:
   from pyproj import Geod
   pyprojFlag=True
except:
   print("Could not import pyproj")
   pyprojFlag=False
from matplotlib.colors import ColorConverter,LinearSegmentedColormap
from scipy.stats.mstats import mquantiles
from irregularInterpolation import interpolationGrid

def discretizeColormap(colmap,N):
   cmaplist = [colmap(i) for i in range(colmap.N)]
   return colmap.from_list('Custom discrete cmap', cmaplist, N)

def hovmoeller(t,dz,Var,contours=10,ztype="dz",orientation="up",**opts):
    if ztype=="dz": #z-variable gives zell thickness, else cell centre
      if dz.ndim==1:
        dz=tile(dz,[t.shape[0],1])
      z=-dz.cumsum(1)+.5*dz
    else:
      z=dz
    if orientation is not "up":
        Var=fliplr(Var)
    if t.ndim==1:
        t=tile(t,[z.shape[1],1]).T
    contourf(t.T,z.T,Var.T,contours,**opts)

def plotMap(lon_0,lat_0,dx,dy,res='i',marble=True,coast=False):
   map=Basemap(projection='lcc',lat_0=lat_0,lon_0=lon_0,
       #llcrnrlon=lonmin,llcrnrlat=latmin,
       #urcrnrlon=lonmax,urcrnrlat=latmax,
       width=dx,height=dy,
       rsphere=(6378137.00,6356752.3142),resolution=res)
   x_0,y_0=map(lon_0,lat_0)
   lonll,latll=map(x_0-.5*dx,y_0-.5*dy,inverse=True)
   lonul,latul=map(x_0-.5*dx,y_0+.5*dy,inverse=True)
   lonlr,latlr=map(x_0+.5*dx,y_0-.5*dy,inverse=True)
   lonur,latur=map(x_0+.5*dx,y_0+.5*dy,inverse=True)
   lonmin=min(lonll,lonul,lonlr,lonur)
   lonmax=max(lonll,lonul,lonlr,lonur)
   latmin=min(latll,latul,latlr,latur)
   latmax=max(latll,latul,latlr,latur)
   if marble:
       im = map.bluemarble()
       map.drawmapboundary()
   else:
       map.drawmapboundary(fill_color=(0./255.,59./255.,80./255.))
       map.fillcontinents(color=(209/255.,162/255.,14/255.,1),lake_color=(0./255.,59./255.,80./255.))
   if coast : map.drawcoastlines()
   #map.drawcountries()
   map.drawmeridians(arange(lonmin,lonmax,(lonmax-lonmin)/5.),labels=[0,0,0,1])
   map.drawparallels(arange(latmin,latmax,(latmax-latmin)/5.),labels=[0,1,0,0])
   return map

def worldMap(center=11.,latres=20.,lonres=20.,marble=True,coast=False):
  m = Basemap(projection='robin',lon_0=center,lat_0=0.,resolution='i')
  if marble:
      im = m.bluemarble()
      m.drawmapboundary()
  else:
      m.drawmapboundary(fill_color=(0./255.,59./255.,80./255.))
      m.fillcontinents(color=(209/255.,162/255.,14/255.,1),lake_color=(0./255.,59./255.,80./255.))
  if coast: m.drawcoastlines(color='k')
  m.drawmeridians(np.arange(ceil((center-180)/lonres)*lonres,ceil(180/lonres)*lonres,lonres),color='0.5',labels=[0,0,0,1])
  m.drawparallels(np.arange(ceil(-90/latres)*latres,ceil(90/latres)*latres,latres),color='0.5',labels=[1,0,0,0])
  return m

def mapVar(map,lon,lat,var):
   #var=v(var)[tlevel,level,:,:].squeeze()
   var=var[:].squeeze()
   x,y = meshgrid(lon,lat)
   x,y = map(x.transpose(),y.transpose())
   pl=map.pcolormesh(x,y,var.transpose())
   return pl

def mapPlace(map,lon,lat,name,marker='ro',xfrac=100.,yfrac=100.,**opts):
    x,y = map(lon,lat)
    map.plot([x],[y],marker)
    text(x+map.xmax/xfrac,y+map.ymax/yfrac,name,**opts)

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

def flipColorMap(cmapStr):
    if type(cmapStr)==type(str()):
      cdata=get_cmap(cmapStr)._segmentdata
    else:
      cdata=cmapStr._segmentdata
      cmapStr=cmapStr.name
    cd={}
    cd['blue']=[(1-line[0],line[1],line[2]) for line in cdata['blue']]
    cd['green']=[(1-line[0],line[1],line[2]) for line in cdata['green']]
    cd['red']=[(1-line[0],line[1],line[2]) for line in cdata['red']]
    cd['red'].reverse()
    cd['blue'].reverse()
    cd['green'].reverse()
    return LinearSegmentedColormap(cmapStr+'Flipped',cd,256)

def discreteColors(noc,cols=['r','b','#FFF000','g','m','c','#FF8000','#400000','#004040','w','b']):
    while noc>len(cols): cols.extend(cols)
    cc=ColorConverter()
    clrs=[cc.to_rgba(col) for col in cols]
    ds=1./float(noc)
    splits=arange(0,1,ds)
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

chlMap=LinearSegmentedColormap('chlMap',
    {'blue':((0.,.1,.1),(.85,.8,.8),(1.,.95,.95)),
    'green':((0.,0.,0.),(.1,0.,0.),(.85,.8,.8),(1.,1.,1.)),
    'red':((0.,0.,0.),(.75,0.,0.),(1.,.5,.5))}, N=256)

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

def pdfMap(fnct):
    b=[(0.,1.,1.),(.05,1.,1.),]
    r=[(0.,1.,1.),(.05,1.,.9),]
    g=[(0.,1.,1.),(.05,1.,.95),]
    N=5
    #for vpos,cpos in zip(arange(.1,1.,.05),arange(1.-1./19.,0.,-1./19.)):
    for n in arange(1,N):
        vpos=.05+n/(N*.95)
        cpos=1.-n/(1.*N)
        fac=fnct(cpos)
        b.append((vpos,fac,fac))
        r.append((vpos,.9*fac,.9*fac))
        g.append((vpos,.95*fac,.95*fac))
    b.append((1.,0.,0.))
    r.append((1.,0.,0.))
    g.append((1.,0.,0.))
    return LinearSegmentedColormap('pdfMap',{'blue':tuple(b),
        'red':tuple(r),'green':tuple(g)})


def pmMap(brightness=.5):
    b=brightness
    cmap=LinearSegmentedColormap('pmMap',
     {'blue':((0.,.0,.0),(.495,.9,.9),(.5,1.,1.),(.505,.9,.9),(1.,.0,.0)),
    'green':((0.,.0,.0),(.495,.9,.9),(.5,1.,1.),(.505,.9,.9),(1.,b,b)),
    'red':((0.,b,b),(.495,.9,.9),(.5,1.,1.),(.505,.9,.9),(1.,.0,.0))}, N=256)
    return cmap

def pmMapDiscrete(clevels=9,brightness=.5):
    b=brightness
    hcl=int(clevels)/2
    cls=[]
    incr=1./hcl
    frac=incr/2.
    for n in range(hcl):
        cls.append((b+(.9-b)*frac,.9*frac,.9*frac))
        frac+=incr
    if clevels%2: cls.append((1.,1.,1.))
    frac=incr/2.
    for n in range(hcl):
        cls.append((.9-.9*frac,.9-(.9-b)*frac,.9-.9*frac))
        frac+=incr
    return discreteColors(clevels,cls)

def brMapDiscrete(clevels=5,brightness=.4):
    b=brightness
    hcl=int(clevels)
    cls=[]
    incr=1./(hcl-1.)
    frac=0.
    for n in range(hcl):
        cls.append([frac,frac,b+(1.-b)*frac]) #shades of blue
        frac+=incr
        frac=min(1,frac)
    frac=0.
    for n in range(hcl):
        cls.append([1.-(1.-b)*frac,1.-frac,1.-frac])  #shades of red
        frac+=incr
        frac=min(1,frac)
    return discreteColors(2*clevels,cls)

def rgMapDiscrete(clevels=5,brightness=.3):
    b=brightness
    hcl=int(clevels)
    cls=[]
    incr=1./(hcl-1.)
    frac=0.
    for n in range(hcl):
        cls.append([b+(1-b)*frac,frac,frac,]) #shades of red
        frac+=incr
        frac=min(1,frac)
    frac=0.
    for n in range(hcl):
        cls.append([1-frac,1-(1.-b)*frac,1-frac,])  #shades of green
        frac+=incr
        frac=min(1,frac)
    return discreteColors(2*clevels,cls)

def mlMap(brightness=.5):
    b=brightness
    cmap=LinearSegmentedColormap('mlMap',
     {'red':((0.,.0,.0),(.495,.75,.75),(.5,1.,1.),(.505,.75,.75),(1.,b,b)),
    'green':((0.,.0,.0),(.495,.75,.75),(.5,1.,1.),(.505,.75,.75),(1.,b,b)),
    'blue':((0.,b,b),(.495,.75,.75),(.5,1.,1.),(.505,.75,.75),(1.,.0,.0))}, N=256)
    return cmap

justifyGrid=lambda x2d,y2d,data,x,y: griddata(x2d.flatten(),y2d.flatten(),data.flatten(),x,y)

def pixeliseGrid(m,x2d,y2d,data,dim,xmin,ymin,xmax,ymax):
    """ pixelise irregular grid data on regular grid """
    (py,px)=dim
    dx=(xmax-xmin)/float(px-1)
    dy=(ymax-ymin)/float(py-1)
    x=linspace(xmin,xmax,px)
    y=linspace(ymin,ymax,py)
    #interpolate data:
    di=justifyGrid(x2d,y2d,data,x,y)
    return di

def mapScalar(bm,data,lonll=-180.,latll=-90.,lonur=180.,latur=90.):
    nx=data.shape[1]
    ny=data.shape[0]
    dlon=(lonur-lonll)/float(nx-1)
    lon=arange(lonll,lonur+dlon/2.,dlon)
    dlat=(latur-latll)/float(ny-1)
    lat=arange(latll,latur+dlat/2.,dlat)
    return bm.transform_scalar(data,lon,lat,nx,ny,checkbounds=False,masked=False)

def getLSMask(m):
    lsm=m.drawlsmask(ocean_color = (0,0,0,255),land_color  = (255,255,255,255),lakes=False).get_array()[:,:,0]/255
    dim=lsm.shape
    #defines land points as 1 == True
    return lsm,dim

def getAxPixel(ax):
    "gets the pixel size of a subplpot axes instance"
    fpx=figPixel(ax.get_figure())
    return (ax.get_position().size*fpx).astype(int)

def convertGridT2U(lon,lat):
    lon2D=zeros([lon.shape[0]+1,lon.shape[1]+1])
    lat2D=zeros([lat.shape[0]+1,lat.shape[1]+1])
    dlon=diff(lon)/2.
    dlat=diff(lat,axis=0)/2.
    lon2D[1:,1:-1]=lon[:,:-1]+dlon
    lat2D[1:-1,1:]=lat[:-1,:]+dlat
    lon2D[1:,0]=lon[:,0]-dlon[:,0]
    lon2D[1:,-1]=lon[:,-1]+dlon[:,-1]
    lat2D[0,1:]=lat[0,:]-dlat[0,:]
    lat2D[-1,1:]=lat[-1,:]+dlat[-1,:]
    dlon=diff(lon2D[1:,:],axis=0)
    lon2D[1:-1,:]=lon2D[1:-1,:]+dlon/2.
    lon2D[0,:]=lon2D[1,:]-dlon[0,:]
    lon2D[-1,:]=lon2D[-2,:]+dlon[-1,:]
    dlat=diff(lat2D[:,1:])
    lat2D[:,1:-1]=lat2D[:,1:-1]+dlat/2.
    lat2D[:,0]=lat2D[:,1]-dlat[:,0]
    lat2D[:,-1]=lat2D[:,-2]+dlat[:,-1]
    return lon2D,lat2D

def pixeliseIrregularData(m,lon2D,lat2D,data):
    lsm,dim=getLSMask(m)
    #lsm=ma.masked_where(lsm==1,lsm,copy=False).mask
    print("Resolution: "+str(dim))
    ax=gca()
    x,y,d=mapData(m,lon2D.ravel(),lat2D.ravel(),data.ravel())
    xfi=linspace(ax.axis()[0],ax.axis()[1],dim[1]+1)
    yfi=linspace(ax.axis()[2],ax.axis()[3],dim[0]+1)
    xci=diff(xfi)/2.+xfi[:-1]
    yci=diff(yfi)/2.+yfi[:-1]
    xfi2D,yfi2D=meshgrid(xfi,yfi)
    #xci2D,yci2D=meshgrid(xci,yci)
    print("interpolating...")
    di=griddata(x,y,d,xci,yci)
    #lon2D=where(lon2D<lonll,lon2D+360.,lon2D)
    #dd=pixeliseGrid(m,lon2D,lat2D,data,dim,lonll,latll,lonur,latur)
    #lon2D=where(lon2D>180,lon2D-360.,lon2D)
    #di=mapScalar(m,dd,lonll,latll,lonur,latur)
    print("Mapped data: "+str(di.shape))
    #di.mask=ma.make_mask(lsm|di.mask)
    return xfi2D,yfi2D,di

figPixel=lambda f: f.get_size_inches()*f.get_dpi()

def mapIrregularData(map,ax,lon,lat,data,xres=2000,yres=1000,mask=False):
    """Maps irregular data on a basemap object using interpolation on a xresxyres grid."""
    x,y,d=mapData(map,lon.ravel(),lat.ravel(),data.ravel(),mask=mask)
    xfi=linspace(ax.axis()[0],ax.axis()[1],xres+1)
    yfi=linspace(ax.axis()[2],ax.axis()[3],yres+1)
    xci=diff(xfi)/2.+xfi[:-1]
    yci=diff(yfi)/2.+yfi[:-1]
    print('\t...interpolating on ',xres,'x',yres,' grid...')
    xfi2D,yfi2D=meshgrid(xfi,yfi)
    #xci2D,yci2D=meshgrid(xci,yci)
    #if mask:
    # Mask=zeros(xci2D.shape)
    # for n in range(len(xci2D.ravel())):
    #	for p in clp:
    #	    if p.contains_point([xci2D.ravel()[n],yci2D.ravel()[n]]):
    #	       Mask.ravel()[n]=(Mask.ravel()[n] or True)
    #	    else:
    #	       Mask.ravel()[n]=(Mask.ravel()[n] or False)
    di=griddata(x,y,d,xci,yci)
    return xfi2D,yfi2D,di

def mapIrregularGrid(map,ax,lon,lat,data,lon0,xres=2000,yres=1000):
    """Maps irregular data on a basemap object using interpolation on a xresxyres grid."""
    lon=masked_where(getmaskarray(data),lon)
    lat=masked_where(getmaskarray(data),lat)
    data=masked_where(getmaskarray(data),getdata(data))
    lon=lon.ravel().compressed()
    lon=where(lon<lon0-180.,lon+360.,lon)
    lon=where(lon>lon0+180.,lon-360.,lon)
    lat=lat.ravel().compressed()
    data=data.ravel().compressed()
    t0=clock()
    T0=time()
    lon,lat,data=removeXYDuplicates(lon,lat,data)
    lonlat=marray([lon,lat]).transpose()
    xfi=linspace(lon0-180.,lon0+180.,xres+1)
    yfi=linspace(-90.,90.,yres+1)
    xci=diff(xfi)/2.+xfi[:-1]
    yci=diff(yfi)/2.+yfi[:-1]
    xfi[0]=xfi[0]+1.e-5
    print('\t...interpolating on ',xres,'x',yres,' grid...')
    xfi2D,yfi2D=meshgrid(xfi,yfi)
    di=griddata(lon,lat,data,xci,yci)
    xci,yci=meshgrid(xci,yci)
    #get position of dateline:
    dl=where(xci[0,:]<-180.,xci[0,:]+360.,xci[0,:]).argmin()
    xfi2D=where(xfi2D>180.,xfi2D-360.,xfi2D)
    xfi2D,yfi2D=map(xfi2D,yfi2D)
    # FOR SOME REASON THIS DOESN'T WORK:?!
    #di=marray(di)
    #di[:,dl:]=maskoceans(xci[:,dl:],yci[:,dl:],di[:,dl:])
    #di[:,:dl]=maskoceans(xci[:,:dl]+360.,yci[:,:dl],di[:,:dl])
    #di.mask=-di.mask
    # SO HAVE TO DO IT THIS WAY
    xci=marray(xci)
    xci[:,dl:]=maskoceans(xci[:,dl:],yci[:,dl:],xci[:,dl:])
    xci[:,:dl]=maskoceans(xci[:,:dl]+360.,yci[:,:dl],xci[:,:dl])
    xci.mask=-xci.mask
    di=masked_where(xci.mask,di)
    return xfi2D,yfi2D,di

def mapIrregular(map,ax,lon,lat,data,lon0,IPT=None,xres=2000,yres=1000):
    """Maps irregular data on a basemap object using interpolation on a xresxyres grid."""
    lon=masked_where(getmaskarray(data),lon)
    lat=masked_where(getmaskarray(data),lat)
    lon=lon.ravel().compress(-getmaskarray(data).ravel())
    lon=where(lon<lon0-180.,lon+360.,lon)
    lon=where(lon>lon0+180.,lon-360.,lon)
    lat=lat.ravel().compress(-getmaskarray(data).ravel())
    data=data.ravel().compress(-getmaskarray(data).ravel())
    lon,lat,data=removeXYDuplicates(lon.data,lat.data,data.data)
    lonlat=marray([lon,lat]).transpose()
    xfi=linspace(lon0-180.,lon0+180.,xres+1)
    yfi=linspace(-90.,90.,yres+1)
    xci=diff(xfi)/2.+xfi[:-1]
    yci=diff(yfi)/2.+yfi[:-1]
    xfi[0]=xfi[0]+1.e-5
    print('\t...interpolating on ',xres,'x',yres,' grid...')
    xfi2D,yfi2D=meshgrid(xfi,yfi)
    #di=griddata(lon.ravel(),lat.ravel(),data.ravel(),xci,yci)
    xci,yci=meshgrid(xci,yci)
    #get position of dateline:
    dl=where(xci[0,:]<-180.,xci[0,:]+360.,xci[0,:]).argmin()
    xci=marray(xci)
    xci[:,dl:]=maskoceans(xci[:,dl:],yci[:,dl:],xci[:,dl:])
    xci[:,:dl]=maskoceans(xci[:,:dl]+360.,yci[:,:dl],xci[:,:dl])
    xci.mask=-xci.mask
    yci=masked_where(xci.mask,yci)
    xciyci=marray([xci.ravel(),yci.ravel()]).transpose()
    if IPT==None:
        IP=interpolationGrid(lonlat,xciyci,Range=0,NoP=3,Mask=getmaskarray(data))
    else:
        IP=IPT
    di=IP(data).reshape([yres,xres])
    xfi2D=where(xfi2D<-180.,xfi2D+360.,xfi2D)
    xfi2D=where(xfi2D>180.,xfi2D-360.,xfi2D)
    xfi2D,yfi2D=map(xfi2D,yfi2D)
    return xfi2D,yfi2D,di,IP

def mapMaskedIrregular(map,ax,lon,lat,data,lon0,lsmask=False,IPT=None,xres=2000,yres=1000):
    """Maps irregular data on a basemap object using interpolation on a xresxyres grid."""
    slmask=logical_not(lsmask)
    lon=getdata(lon).ravel().compress(slmask.ravel())
    lon=where(lon<lon0-180.,lon+360.,lon)
    lon=where(lon>lon0+180.,lon-360.,lon)
    lat=getdata(lat).ravel().compress(slmask.ravel())
    data=data.ravel().compress(slmask.ravel())
    lon,lat,data=removeXYDuplicates(getdata(lon),getdata(lat),data,mask=True)
    lonlat=marray([lon,lat]).transpose()
    xfi=linspace(lon0-180.,lon0+180.,xres+1)
    yfi=linspace(-90.,90.,yres+1)
    xci=diff(xfi)/2.+xfi[:-1]
    yci=diff(yfi)/2.+yfi[:-1]
    xfi[0]=xfi[0]+1.e-5
    print('\t...interpolating on ',xres,'x',yres,' grid...')
    xfi2D,yfi2D=meshgrid(xfi,yfi)
    #di=griddata(lon.ravel(),lat.ravel(),data.ravel(),xci,yci)
    xci,yci=meshgrid(xci,yci)
    #get position of dateline:
    dl=where(xci[0,:]<-180.,xci[0,:]+360.,xci[0,:]).argmin()
    xci=marray(xci)
    xci[:,dl:]=maskoceans(xci[:,dl:],yci[:,dl:],xci[:,dl:])
    xci[:,:dl]=maskoceans(xci[:,:dl]+360.,yci[:,:dl],xci[:,:dl])
    xci.mask=-xci.mask
    yci=masked_where(xci.mask,yci)
    xciyci=marray([xci.ravel(),yci.ravel()]).transpose()
    if IPT==None:
        IP=interpolationGrid(getdata(lonlat),xciyci,Range=0,NoP=3)
    else:
        IP=IPT
    di=IP(data).reshape([yres,xres])
    xfi2D=where(xfi2D<-180.,xfi2D+360.,xfi2D)
    xfi2D=where(xfi2D>180.,xfi2D-360.,xfi2D)
    xfi2D,yfi2D=map(xfi2D,yfi2D)
    return xfi2D,yfi2D,di,IP

def mapMerc(lon,lat,var,marble=False,rivers=False,countries=True,coastlines=False,LandColour=(209/255.,162/255.,14/255.,1),SeaColour=(0./255.,59./255.,80./255.,255./255.),**opts):
   v=var.squeeze()
   dlon=lon.max()-lon.min()
   map = Basemap(projection='merc',llcrnrlon=lon.min(),llcrnrlat=lat.min(),
        urcrnrlon=lon.max(),urcrnrlat=lat.max(),resolution='i')
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor(SeaColour)
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=LandColour)
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=SeaColour)
   xmin=x.min()
   xmax=x.max()
   ymin=y.min()
   ymax=y.max()
   lonmin,latmin=map(xmin,ymin,inverse=True)
   lonmax,latmax=map(xmax,ymax,inverse=True)
   map.drawmeridians(arange(lonmin,lonmax,(lonmax-lonmin)/5.))
   map.drawparallels(arange(latmin,latmax,(latmax-latmin)/5.))
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def mapOrtho(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour=(209/255.,162/255.,14/255.,1),SeaColour=(0./255.,59./255.,80./255.,255./255.),lon_0=False,lat_0=False,**opts):
   if not lon_0: lon_0=lon.mean()
   if not lat_0: lat_0=lat.mean()
   v=var.squeeze()
   map = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='i')
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor(SeaColour)
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=LandColour)
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=SeaColour)
   xmin=x.min()
   xmax=x.max()
   ymin=y.min()
   ymax=y.max()
   map.drawmeridians(arange(-180,180,30))
   map.drawparallels(arange(-90,90,30))
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def mapPlot(lon,lat,var,panning=0.,proj='robin',marble=False,rivers=False,countries=True,coastlines=False,**opts):
   v=var.squeeze()
   panning=panning*1.e6
   dlon=lon.max()-lon.min()
   map = Basemap(projection=proj,lon_0=lon.min()+.5*dlon,rsphere=(6378137.00,6356752.3142),
   	resolution='i')
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=(209/255.,162/255.,14/255.,1))
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=(0./255.,59./255.,80./255.,1))
   xmin=x.min()-panning
   xmax=x.max()+panning
   ymin=y.min()-panning
   ymax=y.max()+panning
   lonmin,latmin=map(xmin,ymin,inverse=True)
   lonmax,latmax=map(xmax,ymax,inverse=True)
   map.drawmeridians(arange(lonmin,lonmax,(lonmax-lonmin)/5.))
   map.drawparallels(arange(latmin,latmax,(latmax-latmin)/5.))
   axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def worldMarble(lon,lat,var,lon0=-160.,countries=True,**opts):
   """Plots a variable given on an irregular grid on a BlueMarble map. Longitude and latitude should be given cell centered."""
   v=var.squeeze()
   map = Basemap(projection='robin',
       lon_0=lon0,
       resolution='i')
   map.drawmapboundary()
   map.drawmeridians(arange(-180,180,30))
   map.drawparallels(arange(-80,80,20.))
   map.drawcoastlines()
   if countries: map.drawcountries()
   xm,ym,dm=pixeliseIrregularData(map,lon,lat,v)
   map.bluemarble()
   p=pcolormesh(xm,ym,dm,**opts)
   return map,p

def worldPlot(lon,lat,data,contours=10,lon0=-160.,xres=None,yres=None,rivers=False,countries=True,marble=False,mask=False,resolution='l',projection="hammer",interp="linear",landcolour="0.7",**opts):
   """Plots a variable given on an irregular grid on a world map using contourf. Longitude and latitude should be given cell centered."""
   if len(lon)==1: lon,lat=meshgrid(lon,lat)
   m = Basemap(projection=projection,
      lon_0=lon0,
      resolution=resolution)
   m.drawmapboundary()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   m.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
   m.drawparallels(arange(-60,61,30.))
   m.drawcoastlines()
   if countries: m.drawcountries()
   if rivers: m.drawrivers(color=(0./255.,59./255.,80./255.,1))
   #Interpolation grid:
   if xres==None: xres=lon.shape[1]
   if yres==None: yres=lon.shape[0]
   dLon=360./xres
   dLat=180./yres
   Lon=arange(lon0-180.+dLon/2.,lon0+180.,dLon)
   Lat=arange(-90.+dLat/2.,90.,dLat)
   Lon,Lat=meshgrid(Lon,Lat)
   X,Y=m(Lon,Lat)
   #Reduce data
   msk=getmaskarray(data)
   if any(msk):
       data=data.ravel().compressed()
       lon=masked_where(msk,lon).ravel().compressed()
       lat=masked_where(msk,lat).ravel().compressed()
   else:
       lon=lon.ravel()
       lat=lat.ravel()
       data=data.ravel()
   lon,lat,data=removeXYDuplicates(lon,lat,data)
   x,y=m(lon,lat)
   #Intepolate:
   Data=griddata(x,y,data,X,Y,interp=interp)
   LSMask=logical_not(maskoceans(Lon,Lat,Data,inlands=False).mask)
   Data=masked_where(LSMask,Data)
   #xi,yi,di=mapIrregularGrid(map,ax,lon,lat,v,lon0,xres=xres,yres=yres)
   if marble:
       m.bluemarble()
       p=m.contourf(X,Y,Data,contours,**opts)
   else:
       m.contourf(X,Y,Data,contours,**opts)
       p=m.fillcontinents(color=landcolour)
   return m,p,X,Y,Data

def worldPlotIPT(lon,lat,var,lon0=-160.,IPT=None,xres=600,yres=400,rivers=False,countries=True,marble=False,mask=False,**opts):
   """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
   v=var.squeeze()
   map = Basemap(projection='robin',
      lon_0=lon0,
      resolution='i')
   map.drawmapboundary()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
   map.drawparallels(arange(-60,61,30.))
   map.drawcoastlines()
   if countries: map.drawcountries()
   if rivers: map.drawrivers(color=(0./255.,59./255.,80./255.,1))
   xi,yi,di,IP=mapIrregular(map,ax,lon,lat,v,lon0,IPT=IPT,xres=xres,yres=yres)
   if marble:
       map.bluemarble()
       p=pcolormesh(xi,yi,di,**opts)
   else:
       p=map.pcolormesh(xi,yi,di,**opts)
       pol=map.fillcontinents(color=(209/255.,162/255.,14/255.,1))
   return map,p,IP

def worldPlotMasked(lon,lat,var,lon0=-160.,IPT=None,xres=600,yres=400,rivers=False,countries=True,marble=False,**opts):
   """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
   v=var.squeeze()
   map = Basemap(projection='robin',
       lon_0=lon0,
       resolution='i')
   map.drawmapboundary()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
   map.drawparallels(arange(-60,61,30.))
   map.drawcoastlines()
   if countries: map.drawcountries()
   if rivers: map.drawrivers(color=(0./255.,59./255.,80./255.,1))
   lsMask=logical_not(maskoceans(lon,lat,getdata(v)).mask)
   xi,yi,di,IP=mapMaskedIrregular(map,ax,lon,lat,v,lon0,lsmask=lsMask,IPT=IPT,xres=xres,yres=yres)
   if marble:
       map.bluemarble()
       p=pcolormesh(xi,yi,di,**opts)
   else:
       p=map.pcolormesh(xi,yi,di,**opts)
       pol=map.fillcontinents(color=(209/255.,162/255.,14/255.,1))
   return map,p,IP

def worldCylPlot(lon,lat,var,lon0=0.,IPT=None,xres=500,yres=250,rivers=False,countries=True,marble=False,grid=True,coastlines=True,mask=False,**opts):
   """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
   v=var.squeeze()
   map = Basemap(projection='cyl',
       lon_0=lon0,
       resolution='i')
   map.drawmapboundary()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   if grid:
       map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
       map.drawparallels(arange(-60,60,30.))
   if coastlines:
       map.drawcoastlines()
   if countries: map.drawcountries()
   if rivers: map.drawrivers(color=(0./255.,59./255.,80./255.,1))
   xi,yi,di=mapIrregularGrid(map,ax,lon,lat,v,lon0,xres=xres,yres=yres)
   if marble:
       map.bluemarble()
       p=pcolormesh(xi,yi,di,**opts)
   else:
       p=map.pcolormesh(xi,yi,di,**opts)
       pol=map.fillcontinents(color=(209/255.,162/255.,14/255.,1))
   return map,p

def mapTMerc(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour=(209/255.,162/255.,14/255.,1),SeaColour=(0./255.,59./255.,80./255.,255./255.),lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
   if not lon_0: lon_0=lon.mean()
   if not lat_0: lat_0=lat.mean()
   latmin=lat.min()
   latmax=lat.max()
   lonmin=lon.min()
   lonmax=lon.max()
   v=var.squeeze()
   map = Basemap(projection='tmerc',lon_0=lon_0,lat_0=lat_0,resolution='i',width=width,height=height)
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor(SeaColour)
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=LandColour)
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=SeaColour)
   xmin=x.min()
   xmax=x.max()
   ymin=y.min()
   ymax=y.max()
   map.drawmeridians(arange(-180,180,dl))
   map.drawparallels(arange(-90,90,dl))
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def mapCassini(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour=(209/255.,162/255.,14/255.,1),SeaColour=(0./255.,59./255.,80./255.,255./255.),lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
   if not lon_0: lon_0=lon.mean()
   if not lat_0: lat_0=lat.mean()
   latmin=lat.min()
   latmax=lat.max()
   lonmin=lon.min()
   lonmax=lon.max()
   v=var.squeeze()
   map = Basemap(projection='cass',lon_0=lon_0,lat_0=lat_0,resolution='i',width=width,height=height)
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor(SeaColour)
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=LandColour)
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=SeaColour)
   xmin=x.min()
   xmax=x.max()
   ymin=y.min()
   ymax=y.max()
   map.drawmeridians(arange(-180,180,dl))
   map.drawparallels(arange(-90,90,dl))
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def mapLambert(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour=(209/255.,162/255.,14/255.,1),SeaColour=(0./255.,59./255.,80./255.,255./255.),lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
   if not lon_0: lon_0=lon.mean()
   if not lat_0: lat_0=lat.mean()
   latmin=lat.min()
   latmax=lat.max()
   lonmin=lon.min()
   lonmax=lon.max()
   v=var.squeeze()
   map = Basemap(projection='laea',lat_ts=lat_0,lon_0=lon_0,lat_0=lat_0,resolution='i',width=width,height=height)
   if marble: im = map.bluemarble()
   x,y = map(lon,lat)
   if coastlines: map.drawcoastlines()
   ax=gca()
   ax.set_axis_bgcolor(SeaColour)
   map.drawmapboundary()
   pl=map.pcolormesh(x,y,v,**opts)
   if not marble:
       map.fillcontinents(color=LandColour)
       if countries: map.drawcountries()
       if rivers: map.drawrivers(color=SeaColour)
   xmin=x.min()
   xmax=x.max()
   ymin=y.min()
   ymax=y.max()
   map.drawmeridians(arange(-180,180,dl))
   map.drawparallels(arange(-90,90,dl))
   zoom=axis()
   del(v,x,y)
   return pl,map,ax,zoom

def worldRegPlot(lon,lat,var,lon0=0.,rivers=False,countries=True,marble=False,mask=False,**opts):
   """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
   v=var.squeeze()
   map = Basemap(projection='cyl',
	lon_0=lon0,
	resolution='i')
   map.drawmapboundary()
   ax=gca()
   ax.set_axis_bgcolor((0./255.,59./255.,80./255.,255./255.))
   map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
   map.drawparallels(arange(-60,60,30.))
   map.drawcoastlines()
   if countries: map.drawcountries()
   if rivers: map.drawrivers(color=(0./255.,59./255.,80./255.,1))
   if marble:
       map.bluemarble()
       p=pcolormesh(lon,lat,var,**opts)
   else:
       p=pcolormesh(lon,lat,var,**opts)
       pol=map.fillcontinents(color=(209/255.,162/255.,14/255.,1))
   return map,p

def mapData(bm,xdata,ydata,data,mask=False):
    x,y=bm(xdata.ravel(),ydata.ravel())
    d=data.ravel()
    return removeXYDuplicates(x,y,d,mask=mask)

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
      print("Longitude, Lattitude and data size don't match:")
      print("  Lon: "+str(len(x))+" Lat: "+str(len(y))+" Data: "+str(len(d)))
      return
    l=len(x)
    duplMask=ones(l)
    if preserveMask:
      xyd=[[xn,yn,n] for xn,yn,n in zip(x,y,arange(l))]
    else:
      xyd=[[xn,yn,dn] for xn,yn,dn in zip(x,y,d)]
    xyd.sort(lambda x,y:cmp(x[:2],y[:2]))
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
        print("Duplicate points removed in x/y map...")
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

def Faces(CC):
    """Computes faces vector (dimension n+1) from regular
    cell centre vector (dimension n)"""
    cc=CC.ravel()
    f=empty(cc.shape[0]+1)
    f[:-1]=cc-.5*diff(cc)[0]
    f[-1]=CC[-1]+.5*diff(CC)[0]
    return f

def bnd2faces(bnd,):
    if len(bnd.shape)==2:
        return append(bnd[:,0],bnd[-1,1])
    else:
        faces=append(bnd[:,:,0],bnd[:,-1,1].reshape([-1,1]),axis=1)
        return append(faces,append(bnd[-1,:,3],bnd[-1,-1,2].reshape([1,1])).reshape([1,-1]),axis=0)

qclim = lambda data,clfun,qrange=[.01,.99]:clfun(mquantiles(data,prob=qrange))

def climits(s,lim=.01):
    """Computes color limits on the base of linear extension of the (lim,1-lim) quantile range to the full range. This produces more equilibrated color limits by oversaturating extreme values."""
    lim*=.5
    cmin,cmax=mquantiles(s,[lim,1.-lim])
    dc=cmax-cmin
    dl=dc/(1.-2.*lim)*lim
    cmin-=lim*(dc+2.*lim)
    cmax+=lim*(dc+2.*lim)
    return cmin,cmax

def quantPlot(data,notch=True,prob=[.01,.05,.95,.99,],**opts):
    b=boxplot(data,notch=notch,sym="",whis=0.,**opts)
    for n,d in enumerate(data):
      lb=b['boxes'][n].get_data()[1][0]
      ub=b['boxes'][n].get_data()[1][5]
      a=mquantiles(d,prob=prob)
      plot([n+1,n+1,],[a[1],lb],'k:',markerfacecolor='b',**opts)
      plot([n+1,n+1,],[ub,a[2]],'k:',markerfacecolor='b',**opts)
      plot([n+1,n+1,],[a[1],a[2]],'kd',markerfacecolor='b',**opts)
      #plot([n+1,],[array(d).min()],'k',marker=6,**opts)
      #plot([n+1,],[array(d).max()],'k',marker=7,**opts)
    return b


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
