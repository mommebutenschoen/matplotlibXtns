from __future__ import print_function

try:
    from mpl_toolkits.basemap import Basemap
    basemap_installed=True
except ImportError:
    print("Basemap module not found. This matplotlibXtns install excludes basemap functionality.")
    basemap_installed=False

if basemap_installed:
  from mpl_toolkits.basemap import interp,maskoceans,shiftgrid,addcyclic
  from numpy import ceil,where,arange,diff,any,logical_not
  from numpy.ma import masked_where,getmaskarray,getdata
  from numpy.ma import array as marray
  from scipy import linspace,meshgrid
  from scipy.interpolate import griddata as sc_griddata
  from matplotlib.pyplot import pcolormesh,axis,gca,text
  from irregularInterpolation import interpolationGrid

  def griddata(x,y,z,xi,yi,interp="nn"):
    """Funtion to map old matplotlib.mlab style griddata calls to
    scipy.interpolate.griddata calls"""
    if interp=="nn":
        method="nearest"
    else:
        method=interp
    X=array((x,y)).T
    XI,YI=meshgrid(xi,yi)
    XY=array((XI.ravel(),YI.ravel())).T
    ZI=sc_griddata(X,z,XY,method=method).reshape(XI.shape)
    return ZI

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
    di=griddata(x,y,d,xci,yci,interp='linear')
    #lon2D=where(lon2D<lonll,lon2D+360.,lon2D)
    #dd=pixeliseGrid(m,lon2D,lat2D,data,dim,lonll,latll,lonur,latur)
    #lon2D=where(lon2D>180,lon2D-360.,lon2D)
    #di=mapScalar(m,dd,lonll,latll,lonur,latur)
    print("Mapped data: "+str(di.shape))
    #di.mask=ma.make_mask(lsm|di.mask)
    return xfi2D,yfi2D,di

  def getLSMask(m):
    lsm=m.drawlsmask(ocean_color = (0,0,0,255),land_color  = (255,255,255,255),lakes=False).get_array()[:,:,0]/255
    dim=lsm.shape
    #defines land points as 1 == True
    return lsm,dim

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
     lon,lat,data=removeXYDuplicates(lon,lat,data)
     lonlat=marray([lon,lat]).transpose()
     xfi=linspace(lon0-180.,lon0+180.,xres+1)
     yfi=linspace(-90.,90.,yres+1)
     xci=diff(xfi)/2.+xfi[:-1]
     yci=diff(yfi)/2.+yfi[:-1]
     xfi[0]=xfi[0]+1.e-5
     print('\t...interpolating on ',xres,'x',yres,' grid...')
     xfi2D,yfi2D=meshgrid(xfi,yfi)
     di=griddata(lon,lat,data,xci,yci,interp="linear")
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
     xci.mask=~xci.mask
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

  def mapMerc(lon,lat,var,marble=False,rivers=False,countries=True,coastlines=False,LandColour="0.3",SeaColour="1.0",**opts):
    v=var.squeeze()
    dlon=lon.max()-lon.min()
    map = Basemap(projection='merc',llcrnrlon=lon.min(),llcrnrlat=lat.min(),
         urcrnrlon=lon.max(),urcrnrlat=lat.max(),resolution='i')
    if marble: im = map.bluemarble()
    x,y = map(lon,lat)
    if coastlines: map.drawcoastlines()
    ax=gca()
    ax.set_facecolor(SeaColour)
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

  def mapOrtho(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour="0.3",SeaColour="1.0",lon_0=False,lat_0=False,**opts):
    if not lon_0: lon_0=lon.mean()
    if not lat_0: lat_0=lat.mean()
    v=var.squeeze()
    map = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='i')
    if marble: im = map.bluemarble()
    x,y = map(lon,lat)
    if coastlines: map.drawcoastlines()
    ax=gca()
    ax.set_facecolor(SeaColour)
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
    ax.set_facecolor((0./255.,59./255.,80./255.,255./255.))
    map.drawmapboundary()
    pl=map.pcolormesh(x,y,v,**opts)
    if not marble:
        map.fillcontinents(color="0.3")
        if countries: map.drawcountries()
        if rivers: map.drawrivers(color="1.0")
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

  def worldPcolormesh(lon,lat,data,contours=10,lon0=0.,xres=None,yres=None,rivers=False,countries=True,marble=False,mask=False,resolution='l',projection="hammer",interp="linear",landcolour="0.3",**opts):
    """Plots a variable given on an irregular grid on a world map using contourf. Longitude and latitude should be given cell centered."""
    if len(lon)==1: lon,lat=meshgrid(lon,lat)
    #transform lon to -180+lon0,180+lon0 interval
    lon0=lon0%360.
    if lon0>180: lon0-=360
    lon=lon%360.
    lon=where(lon>180+lon0,lon-360+lon0,lon)
    lon=where(lon<-180+lon0,lon+360+lon0,lon)
    m = Basemap(projection=projection,
       lon_0=lon0,
       resolution=resolution)
    m.drawmapboundary()
    ax=gca()
    ax.set_facecolor((0./255.,59./255.,80./255.,255./255.))
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
    #transform Lon to -180,180 interval
    Lat=arange(-90.+dLat/2.,90.,dLat)
    Lon,Lat=meshgrid(Lon,Lat)
    Lon_b=arange(lon0-180.,lon0+180+.1*dLon,dLon)
    #transform Lon to -180,180 interval
    Lat_b=arange(-90.,90.+.1*dLat,dLat)
    Lon_b,Lat_b=meshgrid(Lon_b,Lat_b)
    Xb,Yb=m(Lon_b,Lat_b)
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
        p=m.pcolormesh(Xb,Yb,Data,**opts)
    else:
        m.pcolormesh(Xb,Yb,Data,**opts)
        p=m.fillcontinents(color=landcolour)
    return m,p,Xb,Yb,Data

  def worldContourf(lon,lat,data,contours=10,lon0=-160.,xres=None,yres=None,rivers=False,countries=True,marble=False,mask=False,resolution='l',projection="hammer",interp="linear",landcolour="0.3",**opts):
    """Plots a variable given on an irregular grid on a world map using contourf. Longitude and latitude should be given cell centered."""
    if len(lon)==1: lon,lat=meshgrid(lon,lat)
    #transform lon to -180,180 interval
    lon0=lon0%360.
    if lon0>180: lon0-=360
    lon=lon%360.
    lon=where(lon>180,lon-360,lon)
    m = Basemap(projection=projection,
       lon_0=lon0,
       resolution=resolution)
    m.drawmapboundary()
    ax=gca()
    ax.set_facecolor((0./255.,59./255.,80./255.,255./255.))
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
    #transform Lon to -180,180 interval
    Lon=where(Lon>180,Lon-360,Lon)
    Lon=where(Lon<-180,Lon+360,Lon)
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
    ax.set_facecolor("1.0")
    map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
    map.drawparallels(arange(-60,61,30.))
    map.drawcoastlines()
    if countries: map.drawcountries()
    if rivers: map.drawrivers(color="1.0")
    xi,yi,di,IP=mapIrregular(map,ax,lon,lat,v,lon0,IPT=IPT,xres=xres,yres=yres)
    if marble:
       map.bluemarble()
       p=pcolormesh(xi,yi,di,**opts)
    else:
       p=map.pcolormesh(xi,yi,di,**opts)
       pol=map.fillcontinents(color="0.3")
    return map,p,IP

  def worldPlotMasked(lon,lat,var,lon0=-160.,IPT=None,xres=600,yres=400,rivers=False,countries=True,marble=False,**opts):
    """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
    v=var.squeeze()
    map = Basemap(projection='robin',
        lon_0=lon0,
        resolution='i')
    map.drawmapboundary()
    ax=gca()
    ax.set_facecolor("1.0")
    map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
    map.drawparallels(arange(-60,61,30.))
    map.drawcoastlines()
    if countries: map.drawcountries()
    if rivers: map.drawrivers(color="1.0")
    lsMask=logical_not(maskoceans(lon,lat,getdata(v)).mask)
    xi,yi,di,IP=mapMaskedIrregular(map,ax,lon,lat,v,lon0,lsmask=lsMask,IPT=IPT,xres=xres,yres=yres)
    if marble:
        map.bluemarble()
        p=pcolormesh(xi,yi,di,**opts)
    else:
        p=map.pcolormesh(xi,yi,di,**opts)
        pol=map.fillcontinents(color="0.3")
    return map,p,IP

  def worldCylPlot(lon,lat,var,lon0=0.,IPT=None,xres=500,yres=250,rivers=False,countries=True,marble=False,grid=True,coastlines=True,mask=False,**opts):
    """Plots a variable given on an irregular grid on a world map. Longitude and latitude should be given cell centered."""
    v=var.squeeze()
    map = Basemap(projection='cyl',
        lon_0=lon0,
        resolution='i')
    map.drawmapboundary()
    ax=gca()
    ax.set_facecolor("1.0")
    if grid:
        map.drawmeridians(arange(int(lon0-180)/10*10,int(lon0+180)/10*10,30))
        map.drawparallels(arange(-60,60,30.))
    if coastlines:
        map.drawcoastlines()
    if countries: map.drawcountries()
    if rivers: map.drawrivers(color="1.0")
    xi,yi,di=mapIrregularGrid(map,ax,lon,lat,v,lon0,xres=xres,yres=yres)
    if marble:
        map.bluemarble()
        p=pcolormesh(xi,yi,di,**opts)
    else:
        p=map.pcolormesh(xi,yi,di,**opts)
        pol=map.fillcontinents(color="0.3")
    return map,p

  def mapTMerc(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour="0.3",SeaColour="1.0",lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
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
    ax.set_facecolor(SeaColour)
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

  def mapCassini(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour="0.3",SeaColour="1.0",lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
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
    ax.set_facecolor(SeaColour)
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

  def mapLambert(lon,lat,var,marble=False,rivers=False,countries=False,coastlines=True,LandColour="0.3",SeaColour="1.0",lon_0=False,lat_0=False,width=False,height=False,dl=30,**opts):
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
    ax.set_facecolor(SeaColour)
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
    ax.set_facecolor((0./255.,59./255.,80./255.,255./255.))
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
