from matplotlibXtns import mapLambert

def mapMed(lonM,latM,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    mapLambert(lonM,latM,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=15,lat_0=38.,width=4000000,height=2000000,dl=10,**opts)
def mapBaltic(lonB,latB,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    mapLambert(lonB,latB,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=18,lat_0=60,width=1500000,height=1500000,dl=10,**opts)
def mapNWShelf(lonB,latB,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    mapLambert(lonB,latB,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=0.01,lat_0=53.,width=1900000,height=1900000,dl=10,**opts)
def mapNWEurope(lonB,latB,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    mapLambert(lonB,latB,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=-5,lat_0=50.5,width=2500000,height=2500000,dl=10,**opts)

# Atlantic Margin Model
plotFullAMM = lambda lon,lat,v,marble=False,LandColour="0.3",SeaColour="1.0",**opts: mapLambert(lon,lat,v,marble=marble,lon_0=-5.,lat_0=53.,width=2800000,height=3000000,dl=10,LandColour=LandColour,SeaColour=SeaColour,**opts)
plotMercatorAMM = lambda lon,lat,v,marble=False,LandColour="0.3",SeaColour="1.0",**opts: mapMercator(lon,lat,v,marble=marble,dl=10,LandColour=LandColour,SeaColour=SeaColour,**opts)
plotReducedAMM = lambda lon,lat,v,marble=False,LandColour="0.3",SeaColour="1.0": mapLambert(lon,lat,v,marble=marble,lon_0=-6.5,lat_0=54.2,width=2300000,height=2400000,dl=10,LandColour=LandColour,SeaColour=SeaColour)

# North Atlantic
def BASNMap(lonB,latB,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    p,m,a,z=mapLambert(lonB,latB,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=-38,lat_0=30,width=12000000,height=12000000,dl=10,**opts)
    return p,m,a,z
def NATLMap(lonB,latB,v,marble=False,rivers=False,countries=False,coastlines=True,LandColour='.3',SeaColour='1.',**opts):
    p,m,a,z=mapLambert(lonB,latB,v,marble=marble,rivers=rivers,countries=countries,coastlines=coastlines,LandColour=LandColour,SeaColour=SeaColour,lon_0=-52,lat_0=42,width=9500000,height=9500000,dl=10,**opts)
    return p,m,a,z
