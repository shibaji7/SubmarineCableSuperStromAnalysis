import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
import os
from geopy.distance import geodesic
import copy

from geopy.distance import geodesic
from geopy.point import Point

from geographiclib.geodesic import Geodesic

def generate_gc_points_geo(start_lat, start_lon, end_lat, end_lon, spacing_km=10):
    # WGS84 ellipsoid
    geod = Geodesic.WGS84

    # Inverse calculation: gives distance and initial/final azimuths
    inv = geod.Inverse(start_lat, start_lon, end_lat, end_lon)
    total_distance_km = inv['s12'] / 1000  # meters to kilometers
    num_points = int(total_distance_km // spacing_km)

    line = geod.Line(start_lat, start_lon, inv['azi1'])  # create line from start point and initial azimuth

    points = []
    for i in range(num_points + 1):
        s = i * spacing_km * 1000  # distance in meters
        pos = line.Position(s)
        points.append((pos['lat2'], pos['lon2']))

    return points


def calculate_bathymetry_byLITHO1(o, distance_interval=300):
    # 1. Compute 10 points between 2 points
    # 2. Compute water depth for each locations
    # repete steps 1/2 for all points
    geolats, geolongs = [], []
    for xy in o.geometry["coordinates"]:
        lons, lats = np.array(xy)[:, 0], np.array(xy)[:, 1]
        total_distance = geodesic((lats[0], lons[0]), (lats[-1], lons[-1])).km
        if total_distance > distance_interval:
            for i in range(len(lats) - 1):
                td_km = geodesic((lats[i], lons[i]), (lats[i + 1], lons[i + 1])).km
            geolats.extend(lats)
            geolongs.extend(lons)
    d = pd.DataFrame()
    d["geolats"], d["geolongs"] = geolats, geolongs
    d = d.sort_values(by="geolats")
    return d


def calculate_conductive_profiles_with_distance(dp, dpn, base_name="AJC"):
    from scubas.conductivity import ConductivityProfile
    from scubas.datasets import Site

    cp = ConductivityProfile()
    profiles = []
    bin_n = (dpn.geolats, dpn.geolongs)
    for i in range(len(dp) - 1):
        bin_i, bin_j = (
            (dp.geolats.iloc[i], dp.geolongs.iloc[i]),
            (dp.geolats.iloc[i + 1], dp.geolongs.iloc[i + 1]),
        )
        ipts = cp.get_interpolation_points(bin_i, bin_j)
        profile = cp._compile_profile_(ipts)
        profile = Site.init(
            1.0 / profile["resistivity"].to_numpy(dtype=float),
            profile["thickness"].to_numpy(dtype=float) * 1e3,  # Convert to m
            profile["name"],
            "",
            base_name + f"_{i}",
        )
        td_km = geodesic(bin_i, bin_n).km
        profiles.append(
            dict(
                profile=profile,
                bin_i=bin_i,
                bin_j=bin_n,
                td_km=td_km,
                depth=profile.get_thicknesses(0),
            )
        )
    return profiles


def compute_depth_profiles(d):
    from scubas.conductivity import ConductivityProfile

    cp = ConductivityProfile()

    profiles = []
    water_thk = cp.get_water_layer(
        cp.lithosphere_model, (d.geolats.iloc[0], d.geolongs.iloc[0])
    )
    profiles.append(
        {
            "lat": d.geolats.iloc[0],
            "lon": d.geolongs.iloc[0],
            "bathymetry.meters": water_thk,
            "cum_dist_from_00": d.cum_dist_from_00.iloc[0],
        }
    )
    for i in range(len(d) - 1):
        bin_i, bin_j = (
            (d.geolats.iloc[i], d.geolongs.iloc[i]),
            (d.geolats.iloc[i + 1], d.geolongs.iloc[i + 1]),
        )
        ipts = cp.get_interpolation_points(bin_i, bin_j)
        water_thk = cp.get_water_layer(cp.lithosphere_model, ipts) * 1e3  # to meters
        profiles.append(
            {
                "lat": d.geolats.iloc[i + 1],
                "lon": d.geolongs.iloc[i + 1],
                "bathymetry.meters": water_thk,
                "cum_dist_from_00": d.cum_dist_from_00.iloc[i + 1],
            }
        )
    profiles = pd.DataFrame.from_dict(profiles)
    return profiles


def calculate_conductive_profiles(d, base_name="AJC"):
    from scubas.conductivity import ConductivityProfile
    from scubas.datasets import Site

    cp = ConductivityProfile()
    profiles = []
    for i in range(len(d) - 1):
        bin_i, bin_j = (
            (d.geolats.iloc[i], d.geolongs.iloc[i]),
            (d.geolats.iloc[i + 1], d.geolongs.iloc[i + 1]),
        )
        ipts = cp.get_interpolation_points(bin_i, bin_j)
        profile = cp._compile_profile_(ipts)
        profile = Site.init(
            1.0 / profile["resistivity"].to_numpy(dtype=float),
            profile["thickness"].to_numpy(dtype=float) * 1e3,  # Convert to m
            profile["name"],
            "",
            base_name + f"_{i}",
        )
        td_km = geodesic(bin_i, bin_j).km
        profiles.append(
            dict(
                profile=profile,
                bin_i=bin_i,
                bin_j=bin_j,
                td_km=td_km,
                depth=profile.get_thicknesses(0),
            )
        )
    return profiles


def plot_routes(o, geo=None, fname="figures/ajc_routes.png", d=dt.datetime(1958, 2, 11), full_o=None):
    import sys
    sys.path.append("py/")
    from fan import CartoDataOverlay

    cb = CartoDataOverlay(
        date=d,
        central_longitude=130,
        central_latitude=20,
        extent=[110, 170, -50, 50],
        plt_lats=np.arange(-90, 80, 10),
    )
    ax = cb.add_axes()

    if full_o is not None:
        for xy in full_o.geometry["coordinates"]:
            xy = np.array(xy)
            lon, lat = xy[:, 0], xy[:, 1]
            xyz = cb.proj.transform_points(cb.geo, lon, lat)
            ax.plot(xyz[:, 0], xyz[:, 1], ls="--", lw=0.8, color="r", transform=cb.proj)
    
    for xy, c in zip(o.geometry["coordinates"], ["b", "k", "r", "g", "m"]):
        xy = np.array(xy)
        lon, lat = xy[:, 0], xy[:, 1]
        xyz = cb.proj.transform_points(cb.geo, lon, lat)
        ax.plot(xyz[:, 0], xyz[:, 1], ls="-", lw=0.4, color=c, transform=cb.proj)

    if geo is not None:
        xyz = cb.proj.transform_points(
            cb.geo, np.array(geo.geolongs), np.array(geo.geolats)
        )
        ax.plot(xyz[:, 0], xyz[:, 1], ".", ms=0.8, color="b", transform=cb.proj)
    if os.path.exists("data/2024/AJC/20250211-20-39-supermag.csv"):
        o = pd.read_csv("data/2024/AJC/20250211-20-39-supermag.csv", parse_dates=["Date_UTC"])
        iagas = o.IAGA.unique()
        for iaga in iagas:
            x = o[o.IAGA == iaga]
            # print([x["GEOLON"].tolist()[0]], [x["GEOLAT"].tolist()[0]])
            Lon, Lat = [x["GEOLON"].tolist()[0]], [x["GEOLAT"].tolist()[0]]
            xyz = cb.proj.transform_points(cb.geo, np.array(Lon), np.array(Lat))
            ax.scatter(xyz[:, 0], xyz[:, 1], s=4, color="m", marker="D", transform=cb.proj)
    cb.save(fname)
    cb.close()
    return


def get_cable_route(
    url="https://www.submarinecablemap.com/api/v3/cable/cable-geo.json",
    name_key="australia-japan-cable-ajc",
):
    o = None
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        for d in data["features"]:
            if d["properties"]["id"] == name_key:
                o = SimpleNamespace(**d)
    return o


def plot_bathymatry(profiles):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]

    fig = plt.figure(figsize=(5, 2), dpi=300)
    ax = fig.add_subplot(111)
    distance, depths = [], []
    for profile in profiles:
        distance.append(profile["td_km"])
        depths.append(profile["depth"])
    ax.plot(
        np.cumsum(distance),
        np.array(depths) / 1e3,
        ls="-",
        lw=0.8,
        color="r",
    )
    ax.invert_yaxis()
    ax.set_ylim(8, 0)
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Depths, km")
    ax.set_xlim(0, np.cumsum(distance)[-1])
    fig.savefig("figures/ajc_route_bathymetry.png", bbox_inches="tight")
    return

def find_nearby_coordinates(cords, lat=13.3824, lon=144.6973):
    d = [np.sqrt((c[0]-lon)**2+(c[1]-lat)**2) for c in cords]
    d_arg_min = np.argmin(d)
    cords = cords[:d_arg_min]
    return cords

def compute_bathy_profile(points):
    from scubas.conductivity import ConductivityProfile
    
    cp = ConductivityProfile()
    records = []
    records.append({
        "lat":points[0][0], "lon":points[0][1], 
        "bathymetry.meters": cp.get_water_layer(
            cp.lithosphere_model, (points[0][0], points[0][1])
        )*1e3,
        "cum_dist_from_00":0.
    })
    for i in range(1,len(points)):
        bin_i, bin_j = (
            (points[i-1][0], points[i-1][1]),
            (points[i][0], points[i][1]),
        )
        records.append({
            "lat":points[i-1][0], "lon":points[i-1][1], 
            "bathymetry.meters": cp.get_water_layer(
                cp.lithosphere_model, (points[i-1][0], points[i-1][1])
            )*1e3,
            "cum_dist_from_00":geodesic(bin_i, bin_j).km
        })
    records = pd.DataFrame.from_records(records)
    records.cum_dist_from_00 = np.cumsum(records.cum_dist_from_00)
    records.to_csv("data/2024/AJC/lat_long_bathymetry.csv")
    return

def find_AJC_location_by_GUAM():
    o, full_o = (get_cable_route(), get_cable_route())
    cords1 = find_nearby_coordinates(o.geometry["coordinates"][0])
    cords2 = find_nearby_coordinates(o.geometry["coordinates"][2])
    o.geometry["coordinates"] = [cords1] + [cords2]
    d = calculate_bathymetry_byLITHO1(o)
    d.drop_duplicates(inplace=True)
    profiles = calculate_conductive_profiles(d)
    plot_routes(o, d, full_o=full_o)
    points = []
    for p in profiles:
        start, end = p["bin_i"], p["bin_j"]
        points += generate_gc_points_geo(start[0], start[1], end[0], end[1])
    compute_bathy_profile(points)
    plot_bathymatry(profiles)
    return

if __name__ == "__main__":
    find_AJC_location_by_GUAM()
