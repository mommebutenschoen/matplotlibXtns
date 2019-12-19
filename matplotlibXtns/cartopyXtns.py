from __future__ import print_function
try:
    from cartopy import crs
    cartopy_installed=True
except ImportError:
    print("Cartopy module not found. This matplotlibXtns install excludes cartopy functionality.")
    cartopy_installed=False
try:
    from itertools import izip as zip
except ImportError:
    pass
if cartopy_installed:
    from cartopy import feature
    from numpy import arange,array,unique,diff,meshgrid,zeros,logical_or,any,where,ones
    from scipy.interpolate import griddata
    from pdb import set_trace
    from matplotlib.pyplot import figure
    from numpy.ma import getmaskarray,masked_where

    class oceanMap:
        def __init__(self,lon0=0.,prj=crs.PlateCarree,*args,**opts):
            """Class for global maps using cartopy Mollweide or PlateCarree projections.

            Args:
                lon_0(float): central longitude
                prj(string or cartopy.crs object): projection used, currently implemented
                    for Mollweide or PlateCarree (in case of string argument "Mollweide" or "PlateCarree")
                *args,**opts: passed to prj.__init__ function.
            """

            if type(prj)==str:
                if prj.lower().strip()=="mollweide":
                    prj=crs.Mollweide
                elif prj.lower().strip()=="platecarree":
                    prj=crs.PlateCarree
                else:
                    print("{} projection not implemented, using Mollweide instead...".format(prj))
                    prj=crs.Mollweide
            self.prj=prj(lon0,*args,**opts)
            self.ref_prj=crs.PlateCarree()
            #get bounding box of the globe in native coordinates

        def contourf(self,x,y,*args,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            if not f:
                f=figure(figsize=[12,6])
            if not ax:
                ax=f.add_subplot(111,projection=self.prj)
            cnt=ax.contourf(x,y,*args,**opts,zorder=-1)
            ax.coastlines(land_res)
            if land_colour:
                print("Filling continents")
                ax.add_feature(feature.NaturalEarthFeature('physical', 'land', land_res,
                                        facecolor=land_colour))
            #ax.set_extent([xmin,xmax,ymin,ymax])
            if colourbar: cb=f.colorbar(pcm,ax=ax,shrink=.5,aspect=10)
            return cnt,cb

        def pcolormesh(self,x,y,*args,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            if not f:
                f=figure(figsize=[12,6])
            if not ax:
                ax=f.add_subplot(111,projection=self.prj)
            pcm=ax.pcolormesh(x,y,*args,**opts,zorder=-1)
            ax.coastlines(land_res)
            if land_colour:
                ax.add_feature(feature.NaturalEarthFeature('physical', 'land', land_res,
                                        facecolor=land_colour))
            if colourbar: cb=f.colorbar(pcm,ax=ax,shrink=.5,aspect=10)
            return pcm,cb

        def interpolate(self,lon,lat,data,*args,res=360.,bounds=False,zoom=0,mask=False,**opts):
            if zoom:
                xy=self.prj.transform_points(self.ref_prj,lon.ravel(),lat.ravel())
                xmin,xmax=xy[:,0].min(),xy[:,0].max()
                ymin,ymax=xy[:,1].min(),xy[:,1].max()
                Dx=xmin-xmax
                Dy=ymin-ymax
                xmin=xmin-(zoom-100.)/200.*Dx
                xmax=xmax+(zoom-100.)/200.*Dx
                ymax=ymax+(zoom-100.)/200.*Dy
                ymin=ymin-(zoom-100.)/200.*Dy
            else:
                xmin,xmax=self.prj.x_limits
                ymin,ymax=self.prj.y_limits
            dx=(xmax-xmin)/res
            dy=2.*(ymax-ymin)/res
            #regular target coordinates:
            x=arange(xmin+.5*dx,xmax,dx)
            y=arange(ymin+.5*dy,ymax,dy)
            xx,yy=meshgrid(x,y)
            lon=lon.ravel()
            lat=lat.ravel()
            data=data.ravel()
            Mask=getmaskarray(data)
            if any(Mask):
                lon=masked_where(Mask,lon).compressed()
                lat=masked_where(Mask,lat).compressed()
                data=data.compressed()
            (lon,lat),unq_id=unique(array([lon,lat]),return_index=True,axis=1)
            data=data[unq_id]
            xy_trns=self.prj.transform_points(self.ref_prj,lon,lat)
            xin=xy_trns[:,0]
            yin=xy_trns[:,1]
            if len(args)==0 and "method" not in opts.keys():
                opts["method"]="nearest"
            dxy=griddata((xin,yin),data,(xx.ravel(),yy.ravel()),*args,**opts)
            #Mask data outside globe:
            xylonlat=self.ref_prj.transform_points(self.prj,xx.ravel(),yy.ravel())
            globmask=zeros(xx.ravel().shape,bool)
            globmask=where(logical_or.reduce((xylonlat[:,0]>180,xylonlat[:,0]<-180,
                xylonlat[:,1]>90,xylonlat[:,1]<-90)),True,globmask)
            if any(globmask):
                dxy=masked_where(globmask,dxy)
            if mask:
                (xm,ym),unq_id=unique(array([xy[:,0].ravel(),xy[:,1].ravel()]),return_index=True,axis=1)
                uMask=1*Mask[unq_id]
                mxy=griddata((xm,ym),uMask.ravel(),(xx.ravel(),yy.ravel()),*args,**opts)
                dxy=masked_where(mxy,dxy)
            if bounds:
                xb=arange(xmin,xmax+.1*dx,dx)
                yb=arange(ymin,ymax+.1*dy,dy)
                return x,y,dxy.reshape(xx.shape),xb,yb
            else:
                return x,y,dxy.reshape(xx.shape)

        def interpolated_contourf(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,zoom=0,**opts):
            x,y,d=self.interpolate(lon,lat,data,res=res,zoom=zoom)
            print("interpolated coordinate range:",xb.min(),xb.max(),yb.min(),yb.max())
            print("map coordinate range:",self.prj.x_limits,self.prj.y_limits)
            return self.contourf(x,y,d,*args,land_colour=land_colour,f=f,ax=ax,colourbar=colourbar,**opts)

        def interpolated_pcolormesh(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,zoom=0,**opts):
            x,y,d,xb,yb=self.interpolate(lon,lat,data,res=res,bounds=True,zoom=zoom)
            print("interpolated coordinate range:",xb.min(),xb.max(),yb.min(),yb.max())
            print("map coordinate range:",self.prj.x_limits,self.prj.y_limits)
            return self.pcolormesh(xb,yb,d,*args,land_colour=land_colour,f=f,ax=ax,colourbar=colourbar,**opts)

    class globalOceanMap(oceanMap):
        def __init__(self,lon0=0.,prj=crs.Mollweide,*args,**opts):
            oceanMap.__init__(self,lon0=lon0,prj=prj,*args,**opts)

    class regionalOceanMap(oceanMap):
        def __init__(self,lon0=0.,lat0=0.,prj=crs.AlbersEqualArea,*args,**opts):
            if type(prj)==str:
                if prj.lower().strip()=="albersequalarea":
                    prj=crs.AlbersEqualArea
                elif prj.lower().strip()=="platecarree":
                    prj=crs.PlateCarree
                else:
                    print("{} projection not implemented, using AlbersEqualArea instead...".format(prj))
                    prj=crs.AlbersEqualArea
            if array(lon0).ndim>0:
                try:
                    lon0=(lon0.max()-lon0.min())/2.
                except AttributeError:
                    lon0=(max(lon0)-min(lon0))
            if array(lat0).ndim>0:
                try:
                    lat0=(lat0.max()-lat0.min())/2.
                except AttributeError:
                    lat0=(max(lat0)-min(lat0))
            if prj==crs.AlbersEqualArea:
                print("Initialising regional AlbersEqualArea projection centred at {}N,{}E".format(lat0,lon0))
                self.prj=prj(central_longitude=lon0,central_latitude=lat0,*args,**opts)
            else:
                print("Initialising regional PlateCarree projection centred at {}E".format(lon0))
                self.prj=prj(lon0,*args,**opts)
            self.ref_prj=crs.PlateCarree()

        def interpolated_contourf(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,zoom=101,**opts):
            return oceanMap.interpolated_contourf(self,lon,lat,data,*args,res=res,land_colour=land_colour,land_res=land_res,f=f,ax=ax,colourbar=colourbar,zoom=zoom,**opts)

        def interpolated_pcolormesh(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,zoom=101,**opts):
            return oceanMap.interpolated_pcolormesh(self,lon,lat,data,*args,res=res,land_colour=land_colour,land_res=land_res,f=f,ax=ax,colourbar=colourbar,zoom=zoom,**opts)

        def interpolate(self,lon,lat,data,*args,res=360.,bounds=False,zoom=101,mask=True,**opts):
            return oceanMap.interpolate(self,lon,lat,data,*args,res=res,bounds=bounds,zoom=zoom,mask=mask,**opts)

    def mask_feature(x2d,y2d,feat=feature.LAND,eps=1.e-5):
            Mask=ones(x2d.shape,bool)
            for n,(x,y) in enumerate(zip(x2d.ravel(),y2d.ravel())):
                try:
                    next(feat.intersecting_geometries((x-eps,x+eps,y-eps,y+eps)))
                except StopIteration:
                    Mask.ravel()[n]=False
            return Mask
