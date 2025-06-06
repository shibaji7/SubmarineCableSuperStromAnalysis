import os
from types import SimpleNamespace

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from geopy.distance import great_circle as GC  # type: ignore
from loguru import logger

os.makedirs("figures/", exist_ok=True)


def plot_profiles(file_path, segments, colors, plot_file, names):
    bathymetry = BathymetryAnalysis(file_path, segments, colors)
    bathymetry.load_data()
    bathymetry.plot_bathymetry(
        plot_file,
        names=names,
        xticks=[0, 500, 1000, 2000, 4000, 8000],
        xlim=[0, bathymetry.bathymetry_data.distance.iloc[-1] / 1e3],
        ylim=[-8, 0.5],
        yticks=[-8, -6, -4, -2, -1, -0.5, 0],
        yticklabels=[8, 6, 4, 2, 1, 0.5, 0],
    )
    return bathymetry


def get_AJC_segments(gtype="lat"):
    file_path = "data/2024/AJC/lat_long_bathymetry.csv"
    segments = [
        (0, 26),
        (26, 300),
        (300, 410),
        (410, 600),
        (600, 670),
        (670, 780),
        (780, 860),
        (860, 885),
        (885, -1),
    ]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "gold",
        "limegreen",
        "darkviolet",
        "crimson",
        "teal",
        "peru",
        "orchid",
        "slategray",
        "salmon",
        "darkkhaki",
    ]
    # Initialize and use the BathymetryAnalysis class
    bathymetry = BathymetryAnalysis(file_path, segments, colors)
    bathymetry.load_data()
    names = [
        "DO-1",
        "DO-2",
        "DO-3",
        "RDG-1",
        "DO-4",
        "RDG-2",
        "DO-5",
        "DO-6",
        "CS-A",
    ]
    bathymetry.plot_bathymetry(
        "figures/2024/AJC/bathymetry_AJC.png",
        names=names,
        xticks=[0, 500, 1000, 2000, 4000, 8000],
        xlim=[0, bathymetry.bathymetry_data.distance.iloc[-1] / 1e3],
        ylim=[-8, 0.5],
        yticks=[-8, -6, -4, -2, -1, -0.5, 0],
        yticklabels=[8, 6, 4, 2, 1, 0.5, 0],
    )
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
    print(f"Segments>>, {segment_coordinates}")
    return segment_coordinates[:, 0] if gtype == "lat" else segment_coordinates[:, 1]


def get_TAT1_segments(gtype="lat"):
    file_path = "data/1958/lat_long_bathymetry.csv"
    segments = [
        (0, 32),
        (32, 50),
        (50, 60),
        (60, 170),
        (170, 330),
        (330, 410),
        (410, 430),
        (430, -1),
    ]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "gold",
        "limegreen",
        "darkviolet",
        "crimson",
        "teal",
        "peru",
        "orchid",
        "slategray",
        "salmon",
        "darkkhaki",
    ]

    # Initialize and use the BathymetryAnalysis class
    bathymetry = BathymetryAnalysis(file_path, segments, colors)
    bathymetry.load_data()
    bathymetry.plot_bathymetry("figures/bathymetry.png")
    segment_coordinates = np.array(bathymetry.get_segment_coordinates())
    print(f"Segments>>, {segment_coordinates}")
    return segment_coordinates[:, 0] if gtype == "lat" else segment_coordinates[:, 1]


class BathymetryAnalysis:
    """
    A class to handle bathymetry data analysis and visualization.
    """

    def __init__(self, file_path, segments, colors):
        """
        Initialize the BathymetryAnalysis class.

        Parameters:
        - file_path: str, path to the bathymetry dataset (CSV file).
        - segments: list of tuples, index ranges for segmentation.
        - colors: list of str, colors for plotting each segment.
        """
        self.file_path = file_path
        self.segments = segments
        self.colors = colors
        self.bathymetry_data = None
        self.segment_coordinates = []
        return

    def load_data(self):
        """
        Load bathymetry data from the CSV file.
        """
        self.bathymetry_data = pd.read_csv(self.file_path)
        self.bathymetry_data["distance"] = 0.0
        for i in range(1, len(self.bathymetry_data)):
            # Calculate distance using geopy's great_circle function
            self.bathymetry_data.loc[i, "distance"] = GC(
                (
                    self.bathymetry_data["lat"].iloc[i - 1],
                    self.bathymetry_data["lon"].iloc[i - 1],
                ),
                (
                    self.bathymetry_data["lat"].iloc[i],
                    self.bathymetry_data["lon"].iloc[i],
                ),
            ).meters
            # Calculate cumulative distance
            self.bathymetry_data.loc[i, "distance"] += self.bathymetry_data[
                "distance"
            ].iloc[i - 1]
        return

    def plot_bathymetry(
        self,
        output_path,
        dpi=1000,
        figsize=(8, 3),
        names=[],
        xlim=[0, 4000],
        xticks=[0, 500, 2000, 4000],
        ylim=[-5, 0.5],
        yticks=[-5, -4, -3, -2, -1, -0.5],
        yticklabels=[5, 4, 3, 2, 1, 0.5],
    ):
        """
        Plot the bathymetry data with segments and save the figure.

        Parameters:
        - output_path: str, path to save the output figure.
        - dpi: int, resolution of the output figure.
        - figsize: tuple, size of the figure.
        """
        if self.bathymetry_data is None:
            raise ValueError("Bathymetry data not loaded. Call load_data() first.")

        fig, ax = plt.subplots(dpi=dpi, figsize=figsize, nrows=1, ncols=1)

        # Plot the full bathymetry data
        ax.plot(
            self.bathymetry_data.distance / 1e3,
            -1 * self.bathymetry_data["bathymetry.meters"] / 1e3,
            color="k",
            lw=0.6,
        )

        dist, depth = [], []
        # Plot each segment with a different color
        for i, seg in enumerate(self.segments):
            segment_data = self.bathymetry_data.iloc[seg[0] : seg[1]]
            dist.append(segment_data.distance.tolist()[0] / 1e3)
            depth.append(segment_data["bathymetry.meters"].mean() / 1e3)
            ax.plot(
                segment_data.distance / 1e3,
                -1 * segment_data["bathymetry.meters"] / 1e3,
                marker=".",
                ls="None",
                ms=1.2,
                color=self.colors[i],
            )
            if len(names) == len(self.segments):
                ax.text(
                    segment_data.distance.mean() / 1e3,
                    -(segment_data["bathymetry.meters"].mean() / 1e3) - 0.1,
                    names[i],
                    ha="center",
                    va="top",
                    rotation=90,
                    fontdict=dict(size=10, color="b"),
                )

            # Print initial and final coordinates of the segment
            logger.info(
                f"Initial {segment_data.lat.round(2).iloc[0]},{segment_data.lon.round(2).iloc[0]}"
            )
            logger.info(
                f"Final {segment_data.lat.round(2).iloc[-1]},{segment_data.lon.round(2).iloc[-1]}"
            )

            # Store segment coordinates
            self.segment_coordinates.append(
                [segment_data.lat.round(2).iloc[0], segment_data.lon.round(2).iloc[0]]
            )
        dist.append(self.bathymetry_data.distance.iloc[-1] / 1e3)
        depth.append(self.bathymetry_data["bathymetry.meters"].iloc[-1] / 1e3)
        depth = np.array(depth)
        depth[depth > 0] = depth[depth > 0] * -1
        ax.step(
            dist,
            depth,
            where="post",
            ls="-",
            lw=1.5,
            color="k",
        )

        # Append the final coordinate of the last segment
        self.segment_coordinates.append(
            [segment_data.lat.round(2).iloc[-1], segment_data.lon.round(2).iloc[-1]]
        )

        # Customize plot appearance
        ax.set_xticks(xticks)
        ax.set_xlabel("Distance, km")
        ax.set_xlim(xlim)
        ax.axhline(0, ls="--", lw=0.4, color="b", alpha=0.7)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel("Depths, km")

        # Save the figure
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return

    def get_segment_coordinates(self):
        """
        Get the coordinates of the segments.

        Returns:
        - list of lists, each containing [latitude, longitude] of segment boundaries.
        """
        return self.segment_coordinates


SubSeaCables = SimpleNamespace(
    **dict(
        TAT8=dict(
            Latitudes=[
                39.6,
                38.79,
                37.11,
                39.80,
                40.81,
                43.15,
                44.83,
                46.51,
                47.85,
                50.79,
            ],
            Longitudes=[
                -74.33,
                -72.62,
                -68.94,
                -48.20,
                -45.19,
                -39.16,
                -34.48,
                -22.43,
                -9.05,
                -4.55,
            ],
        ),
        TAT1=dict(Latitudes=get_TAT1_segments(), Longitudes=get_TAT1_segments("lon")),
        AJC=dict(Latitudes=get_AJC_segments(), Longitudes=get_AJC_segments("lon")),
    )
)

# Example usage
if __name__ == "__main__":
    # Define input parameters
    get_TAT1_segments()
