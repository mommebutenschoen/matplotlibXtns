try:
    from cartopy import crs
    cartopy_installed=True
except ImportError:
    print("Cartopy module not found. This matplotlibXtns install excludes cartopy functionality.")
    cartopyt_installed=False
if cartopy_installed:
    from cartopy import feature
    from numpy import arange,array,unique,diff,meshgrid,zeros,logical_or,any,where
    from scipy.interpolate import griddata
    from pdb import set_trace
    from matplotlib.pyplot import figure
    from numpy.ma import getmaskarray,masked_where
    class globalOceanMap:
        def __init__(self,lon0=0.,prj=crs.Mollweide,*args,**opts):
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

        def contourf(self,*args,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            if not f:
                f=figure(figsize=[12,6])
            if not ax:
                ax=f.add_subplot(111,projection=self.prj)
            cnt=ax.contourf(*args,**opts,zorder=-1)
            ax.coastlines(land_res)
            if land_colour:
                print("Filling continents")
                ax.add_feature(feature.LAND, facecolor=land_colour)
            if colourbar: cb=f.colorbar(cnt,ax=ax,shrink=.5,aspect=10)
            return cnt,cb

        def pcolormesh(self,*args,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            if not f:
                f=figure(figsize=[12,6])
            if not ax:
                ax=f.add_subplot(111,projection=self.prj)
            cnt=ax.pcolormesh(*args,**opts,zorder=-1)
            ax.coastlines(land_res)
            if land_colour:
                print("Filling continents")
                ax.add_feature(feature.LAND, facecolor=land_colour)
            if colourbar: cb=f.colorbar(cnt,ax=ax,shrink=.5,aspect=10)
            return cnt,cb

        def interpolate(self,lon,lat,data,*args,res=360.,bounds=False,**opts):
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
            if bounds:
                xb=arange(xmin,xmax+.1*dx,dx)
                yb=arange(ymin,ymax+.1*dy,dy)
                return x,y,dxy.reshape(xx.shape),xb,yb
            else:
                return x,y,dxy.reshape(xx.shape)

        def interpolated_contourf(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            x,y,d=self.interpolate(lon,lat,data,res=res)
            return self.contourf(x,y,d,*args,land_colour=land_colour,f=f,ax=ax,colourbar=colourbar,**opts)

        def interpolated_pcolormesh(self,lon,lat,data,*args,res=360.,land_colour="#485259",land_res='50m',f=False,ax=False,colourbar=True,**opts):
            x,y,d,xb,yb=self.interpolate(lon,lat,data,res=res,bounds=True)
            return self.pcolormesh(xb,yb,d,*args,land_colour=land_colour,f=f,ax=ax,colourbar=colourbar,**opts)
