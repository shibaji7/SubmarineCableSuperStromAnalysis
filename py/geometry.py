#!/usr/bin/env python

"""
geometry.py: 
    This module contains the CartoBase class, which is a subclass of GeoAxes
    from the cartopy library. The CartoBase class is used to create a map projection
    and overlay coastlines and lakes on the map. The class also provides methods to
    add features to the map, such as coastlines, lakes, and oceans. The class
    also provides a method to set the size of the plot. The class is used to
    create a map projection and overlay coastlines and lakes on the map. The class
    also provides methods to add features to the map, such as coastlines, lakes,
    and oceans. The class also provides a method to set the size of the plot.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import os
import sys
sys.path.append("py/")
from bathymetry import SubSeaCables

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from descartes import PolygonPatch
from matplotlib.projections import register_projection
from shapely.geometry import LineString, MultiLineString, Polygon, mapping


def setsize(size=6):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    return


def date_string(date, label_style="web"):
    # Set the date and time formats
    dfmt = "%d %b %Y" if label_style == "web" else "%d %b %Y,"
    tfmt = "%H:%M"
    stime = date
    date_str = "{:{dd} {tt}} UT".format(stime, dd=dfmt, tt=tfmt)
    return date_str


class CartoBase(GeoAxes):
    name = "CartoBase"

    def __init__(self, *args, **kwargs):
        if "map_projection" in kwargs:
            map_projection = kwargs.pop("map_projection")
        else:
            map_projection = cartopy.crs.NorthPolarStereo()
            print(
                "map_projection keyword not set, setting it to cartopy.crs.NorthPolarStereo()"
            )
        # first check if datetime keyword is given!
        # it should be since we need it for aacgm
        if "plot_date" in kwargs:
            self.plot_date = kwargs.pop("plot_date")
        else:
            raise TypeError(
                "need to provide a date using 'plot_date' keyword for aacgmv2 plotting"
            )
        # Now work with the coords!
        supported_coords = ["geo", "aacgmv2", "aacgmv2_mlt"]
        if "coords" in kwargs:
            self.coords = kwargs.pop("coords")
            if self.coords not in supported_coords:
                err_str = "coordinates not supported, choose from : "
                for _n, _sc in enumerate(supported_coords):
                    if _n + 1 != len(supported_coords):
                        err_str += _sc + ", "
                    else:
                        err_str += _sc
                raise TypeError(err_str)
        else:
            self.coords = "geo"
            print("coords keyword not set, setting it to aacgmv2")
        # finally, initialize te GeoAxes object
        super().__init__(map_projection=map_projection, *args, **kwargs)
        return

    def overaly_coast_lakes(self, resolution="50m", color="black", **kwargs):
        """
        Overlay AACGM coastlines and lakes
        """
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        # overaly coastlines
        feature = cartopy.feature.NaturalEarthFeature(
            "physical", "land", resolution, edgecolor="face", facecolor="lightgray"
        )
        self.add_feature(cartopy.feature.COASTLINE, **kwargs)
        # self.add_feature(feature)
        # ax.coastlines(resolution=resolution)

    def add_feature(self, feature, **kwargs):
        # Now we"ll set facecolor as None because aacgm doesn"t close
        # continents near equator and it turns into a problem
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        if "facecolor" in kwargs:
            print(
                "manually setting facecolor keyword to none as aacgm fails for fill! want to know why?? think about equator!"
            )
        kwargs["facecolor"] = "none"
        if self.coords == "geo":
            super().add_feature(feature, **kwargs)
        else:
            aacgm_geom = self.get_aacgm_geom(feature)
            aacgm_feature = cartopy.feature.ShapelyFeature(
                aacgm_geom, cartopy.crs.Geodetic(), **kwargs
            )
            super().add_feature(aacgm_feature, **kwargs)

    def mark_latitudes(self, lat_arr, lon_location=-40, **kwargs):
        """
        mark the latitudes
        Write down the latitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        """
        if isinstance(lat_arr, list):
            lat_arr = numpy.array(lat_arr)
        else:
            if not isinstance(lat_arr, numpy.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        # make an array of lon_location
        lon_location_arr = numpy.full(lat_arr.shape, lon_location)
        proj_xyz = self.projection.transform_points(
            cartopy.crs.PlateCarree(), lon_location_arr, lat_arr
        )
        # plot the lats now!
        out_extent_lats = False
        for _np, _pro in enumerate(proj_xyz[..., :2].tolist()):
            # check if lats are out of extent! if so ignore them
            lat_lim = self.get_extent(crs=cartopy.crs.Geodetic())[2::]
            if (lat_arr[_np] >= min(lat_lim)) and (lat_arr[_np] <= max(lat_lim)):
                self.text(
                    _pro[0],
                    _pro[1],
                    r"$%s^{\circ}$" % str(lat_arr[_np]),
                    **kwargs,
                    alpha=0.5,
                )
            else:
                out_extent_lats = True
        if out_extent_lats:
            print("some lats were out of extent ignored them")

    def mark_longitudes(self, lon_arr=numpy.arange(-180, 180, 60), **kwargs):
        """
        mark the longitudes
        Write down the longitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        This is also trickier compared to latitudes!
        """
        if isinstance(lon_arr, list):
            lon_arr = numpy.array(lon_arr)
        else:
            if not isinstance(lon_arr, numpy.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        # get the boundaries
        [x1, y1], [x2, y2] = self.viewLim.get_points()
        bound_lim_arr = []
        right_bound = LineString(([-x1, y1], [x2, y2]))
        top_bound = LineString(([x1, -y1], [x2, y2]))
        bottom_bound = LineString(([x1, y1], [x2, -y2]))
        left_bound = LineString(([x1, y1], [-x2, y2]))
        plot_outline = MultiLineString(
            [right_bound, top_bound, bottom_bound, left_bound]
        )
        # get the plot extent, we"ll get an intersection
        # to locate the ticks!
        plot_extent = self.get_extent(cartopy.crs.Geodetic())
        line_constructor = lambda t, n, b: numpy.vstack(
            (numpy.zeros(n) + t, numpy.linspace(b[2], b[3], n))
        ).T
        for t in lon_arr[:-1]:
            try:
                xy = line_constructor(t, 30, plot_extent)
                # print(xy)
                proj_xyz = self.projection.transform_points(
                    cartopy.crs.Geodetic(), xy[:, 0], xy[:, 1]
                )
                xyt = proj_xyz[..., :2]
                ls = LineString(xyt.tolist())
                locs = plot_outline.intersection(ls)
                if not locs:
                    continue
                # we need to get the alignment right
                # so get the boundary closest to the label
                # and plot it!
                closest_bound = min(
                    [
                        right_bound.distance(locs),
                        top_bound.distance(locs),
                        bottom_bound.distance(locs),
                        left_bound.distance(locs),
                    ]
                )
                if closest_bound == right_bound.distance(locs):
                    ha = "left"
                    va = "top"
                elif closest_bound == top_bound.distance(locs):
                    ha = "left"
                    va = "bottom"
                elif closest_bound == bottom_bound.distance(locs):
                    ha = "left"
                    va = "top"
                else:
                    ha = "right"
                    va = "top"
                if self.coords == "aacgmv2_mlt":
                    marker_text = str(int(t / 15.0))
                else:
                    marker_text = r"$%s^{\circ}$" % str(t)
                self.text(
                    locs.bounds[0] + 0.02 * locs.bounds[0],
                    locs.bounds[1] + 0.02 * locs.bounds[1],
                    marker_text,
                    ha=ha,
                    va=va,
                    **kwargs,
                    alpha=0.5,
                )
            except:
                pass
        return


register_projection(CartoBase)


def create_bathymetrymap(cables=["TAT1"], colors=["m", "gold"], date=dt.datetime(1989, 3, 12)):
    ##############################################################
    # Download GEBCO data from https://www.gebco.net/data_and_products/gridded_bathymetry_data/
    #               and save it in the data/GEBCO_2024/ directory
    # The GEBCO_2014_2D.nc file should be in the data/GEBCO_2024/ directory
    # The GEBCO_2014_2D.nc file contains the bathymetry data
    ##############################################################

    # Load GEBCO data
    path_root = os.getcwd()
    path_data = os.path.join(path_root, "data/GEBCO_2024/")
    da = xr.open_dataset(os.path.join(path_data, "GEBCO_2024.nc"))

    extent = [-80, -5, 20, 60]
    darray = 20
    ## matplotlib work by initialising a matplotlib figure
    ## then you define additional attributes to the figure, like adding data, labels, colors, whatever

    ## initialize a matplotlib figure
    fig = plt.figure(figsize=(4, 4), dpi=300)

    proj = cartopy.crs.Stereographic(
        central_longitude=-70,
        central_latitude=0.30,
    )
    # this creats a 'geoaxes' object and sets the projection to a cool looking orthographic projection
    ax = fig.add_subplot(
        111,
        projection="CartoBase",
        map_projection=proj,
        coords="geo",
        plot_date=date,
    )

    # set the extent of the plot to a global view
    plt_lons = np.arange(-180, 181, 15)
    mark_lons = np.arange(extent[0], extent[1], 10)
    plt_lons = np.arange(-180, 181, 20)
    mark_lons = np.arange(extent[0], extent[1], 20)
    plt_lats = np.arange(extent[1], extent[2], 10)
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.2)
    gl.xlocator = mticker.FixedLocator(plt_lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.n_steps = 90
    ax.mark_latitudes(plt_lats, fontsize="small", color="red")
    ax.mark_longitudes(mark_lons, fontsize="small", color="red")
    data = da["elevation"][::darray, ::darray]
    data = np.ma.masked_where(data >= 0, data)
    im = ax.pcolormesh(
        da["lon"][::darray],
        da["lat"][::darray],
        data/1e3,
        shading="auto",
        cmap="Blues_r",
        transform=ccrs.PlateCarree(),
        vmax=1,
        vmin=-5,
    )
    cax = fig.add_axes([0.15, -0.01, 0.3, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Bathymetry, km")
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax.overaly_coast_lakes(lw=0.4, alpha=0.4)
    ax.add_feature(cartopy.feature.LAND, facecolor="lightgray", lw=0.4)
    ax.text(
        -0.02,
        0.99,
        "Coord: Geo",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=12,
        rotation=90,
    )
    ax.text(
        0.05,
        1.05,
        (f"{date_string(date)}"),
        ha="left",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
    )
    for cbl, color  in zip(cables, colors):
        cable = getattr(
            SubSeaCables, cbl
        )
        ax.scatter(
            cable["Longitudes"],
            cable["Latitudes"], 
            marker="s", s=5,
            c=color,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(
            cable["Longitudes"],
            cable["Latitudes"],
            ls="-", lw=1.2, color="k",
            transform=ccrs.PlateCarree(),
        )
    plt.savefig(
        os.path.join("figures", "GEBCO_2024_Bathymetry.png"), dpi=300, bbox_inches="tight"
    )

if __name__ == "__main__":
    create_bathymetrymap(["TAT1", "TAT8"])