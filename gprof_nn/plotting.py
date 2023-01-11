"""
=================
gprof_nn.plotting
=================

Utility functions for plotting.
"""
import pathlib

import cartopy.crs as ccrs
from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba, to_hex, LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import xarray as xr



_STYLE_FILE = pathlib.Path(__file__).parent / "files" / "matplotlib_style.rc"


def set_style(latex=False):
    """
    Sets matplotlib style to a style file that I find visually more pleasing
    then the default settings.

    Args:
        latex: Whether or not to use latex to render text.
    """
    plt.style.use(str(_STYLE_FILE))
    rc("text", usetex=latex)


def scale_bar(
        ax,
        length,
        location=(0.5, 0.05),
        linewidth=3,
        height=0.01,
        border=0.05,
        parts=4,
        zorder=50,
        textcolor="k"
):
    """
    Draw a scale bar on a cartopy map.

    Args:
        ax: The matplotlib.Axes object to draw the axes on.
        length: The length of the scale bar in meters.
        location: A tuple ``(h, w)`` defining the fractional horizontal
            position ``h`` and vertical position ``h`` in the given axes
            object.
        linewidth: The width of the line.
    """
    import cartopy.crs as ccrs
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    lon_c = lon_min + (lon_max - lon_min) * location[0]
    lat_c = lat_min + (lat_max - lat_min) * location[1]
    transverse_merc = ccrs.TransverseMercator(lon_c, lat_c)

    x_min, x_max, y_min, y_max = ax.get_extent(transverse_merc)

    x_c = x_min + (x_max - x_min) * location[0]
    y_c = y_min + (y_max - y_min) * location[1]

    x_left = x_c - length / 2
    x_right = x_c  + length / 2

    def to_axes_coords(point):
        crs = ax.projection
        p_data = crs.transform_point(*point, src_crs=transverse_merc)
        return ax.transAxes.inverted().transform(ax.transData.transform(p_data))

    def axes_to_lonlat(point):
        p_src = ax.transData.inverted().transform(ax.transAxes.transform(point))
        return ccrs.PlateCarree().transform_point(*p_src, src_crs=ax.projection)


    left_ax = to_axes_coords([x_left, y_c])
    right_ax = to_axes_coords([x_right, y_c])

    l_ax = right_ax[0] - left_ax[0]
    l_part = l_ax / parts



    left_bg = [
        left_ax[0] - border,
        left_ax[1] - height / 2 - border
    ]

    background = Rectangle(
        left_bg,
        l_ax + 2 * border,
        height + 2 * border,
        facecolor="none",
        transform=ax.transAxes,
        zorder=zorder
    )
    ax.add_patch(background)

    for i in range(parts):
        left = left_ax[0] + i * l_part
        bottom = left_ax[1] - height / 2

        color = "k" if i % 2 == 0 else "w"
        rect = Rectangle(
            (left, bottom),
            l_part,
            height,
            facecolor=color,
            edgecolor="k",
            transform=ax.transAxes,
            zorder=zorder
        )
        ax.add_patch(rect)

    x_bar = [x_c - length / 2, x_c + length / 2]
    x_text = 0.5 * (left_ax[0] + right_ax[0])
    y_text = left_ax[1] + 0.5 * height + 2 * border
    ax.text(x_text,
            y_text,
            f"{length / 1e3:g} km",
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='top',
            color=textcolor
    )


def set_cartopy_ticks(ax, left, bottom):
    """
    Add latitude and longitude ticks to cartopy map.

    Args:
        ax: The matplotlib Axes object to add the ticks to.
        left: Whether to add ticks to the left spine
        bottom: Whether to add ticks to the bottom spine
    """
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,
        color='gray',
        alpha=0.5,
    )
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabels_bottom = bottom
    gl.ylabels_left = left


def create_equidistant_area(lon_center, lat_center, extent=2e6, resolution=4e3):
    """
    Creates an area definition for an equidistant projection around a given
    center.

    Args:
        lon_center: The central longitude.
        lat_center: The central latitude.
        extent: The extent of the scene in m
        resolution: The resolution of the scene in m.

    Return:
        An area definition object.
    """
    from pyresample.geometry import AreaDefinition
    area_id = 'GPM scene'
    description = 'Equidistant projection'
    proj_id = 'GPM scene'
    projection = {'proj': 'laea', 'lat_0': lat_center, 'lon_0': lon_center, 'a': 6371228.0, 'units': 'm'}
    width = int(extent / resolution)
    height = int(extent / resolution)
    area_extent = (-0.5 * extent, -0.5 * extent, 0.5 * extent, 0.5 * extent)
    area = AreaDefinition(
        area_id,
        description,
        proj_id,
        projection,
        width,
        height,
        area_extent
    )
    return area

def make_goes_background(time, area, filename, night=False):
    from pansat.products.satellite.goes import GOES16L1BRadiances
    from satpy import Scene

    if not night:
        goes = GOES16L1BRadiances("F", [1, 2, 3])
        composite = "true_color"
    else:
        goes = GOES16L1BRadiances("F", [13])
        composite = "C13"
    goes_files = goes.download(time, time)

    print([str(f) for f in goes_files])
    scene = Scene([str(f) for f in goes_files], reader="abi_l1b")

    scene.load([composite])
    scene_r = scene.resample(area)
    scene_r.load([composite])
    scene_r.save_dataset(composite, filename)


def add_surface(scene, area, background):
    """
    Add surface with background texture to scene.

    Args:
        scene: pyvista Plotter object to which to add the surface with background
            texture.
        background: PIL.Image object containing the background image.
        extent: The extent of the scene in meters.
        resolution: The resolution of the scene in meters.
    """
    import pyvista as pv

    extent = area.area_extent
    m, n = area.shape
    y = np.linspace(extent[0], extent[2], m)
    x = np.linspace(extent[1], extent[3], n)

    z = np.linspace(0, 20, 41) * 1e3
    z = 0.5 * (z[1:] + z[:-1])
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    g = pv.StructuredGrid(
        x[:, :, :2],
        y[:, :, :2],
        z[:, :, :2] * 10
    )
    g.texture_map_to_plane(inplace=True)
    texture = pv.numpy_to_texture(np.array(background)[:, :, ...,:3])
    scene.add_mesh(g, texture=texture)


def add_hydrometeors(
        scene,
        area,
        input_file,
        opacity=None,
        bars=False
):
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_gauss
    import pyvista as pv

    data = xr.open_dataset(input_file)

    lats = data.latitude.data
    lons = data.longitude.data
    swath = SwathDefinition(lats=lats, lons=lons)

    sp = data.surface_precip
    rwc = data.rain_water_content.data
    swc = data.snow_water_content.data

    n_layers = swc.shape[-1]
    rwc_r = resample_gauss(swath, rwc, area, 8e3, fill_value=np.nan, sigmas=[5e3] * n_layers)
    rwc_r = rwc_r
    swc_r = resample_gauss(swath, swc, area, 8e3, fill_value=np.nan, sigmas=[5e3] * n_layers)
    swc_r = swc_r

    extent = area.area_extent
    m, n = area.shape
    y = np.linspace(extent[0], extent[2], m)
    x = np.linspace(extent[1], extent[3], n)
    if n_layers == 28:
        z = np.concatenate([np.linspace(0, 9.5e3, 20) + 0.25e3, 10.5e3 + np.arange(0, 8) * 1e3])
    else:
        z = np.linspace(0, 20, 41) * 1e3
        z = 0.5 * (z[1:] + z[:-1])
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1)

    print(swc_r.shape, xyz.shape)

    g = pv.StructuredGrid(
        xyz[:, :, ..., 0],
        xyz[:, :, ..., 1],
        xyz[:, :, ..., 2] * 10
    )
    norm = LogNorm(1e-2, 1e0)

    m = ScalarMappable(norm=norm, cmap="Reds")
    cm = lambda x: m.to_rgba(x)
    g.point_data["rwc"] = rwc_r[::-1].flatten(order="F")
    rwc_levels = np.linspace(0, 1, 11)[1:]
    colors = [to_hex(c) for c in m.to_rgba(rwc_levels)]
    contours = g.contour(scalars="rwc", isosurfaces=rwc_levels)


    scalar_bar_args = {
        "title": "RWC [g / m³]",
        "position_x": 0.05,
        "position_y": 0.25,
        "vertical": True
    }
    scene.add_mesh(
        contours,
        cmap=colors,
        opacity=opacity,
        show_scalar_bar=bars,
        scalar_bar_args=scalar_bar_args,
        clim=[0.1, 1.0],
        smooth_shading=True,
        ambient=0.35,
        specular=1.0
    )

    g.point_data["swc"] = swc_r[::-1].flatten(order="F")
    swc_levels = np.linspace(0, 1, 11)[1:]
    if opacity is None:
        opacity = "linear"
    m = ScalarMappable(norm=norm, cmap="Blues")
    cm = lambda x: m.to_rgba(x)
    colors = [to_hex(c) for c in m.to_rgba(swc_levels)]

    scalar_bar_args = {
        "title": "SWC [g / m³]",
        "position_x": 0.90,
        "position_y": 0.25,
        "vertical": True
    }
    contours = g.contour(scalars="swc", isosurfaces=swc_levels)
    scene.add_mesh(
        contours,
        cmap=colors,
        opacity=opacity,
        show_scalar_bar=bars,
        scalar_bar_args=scalar_bar_args,
        clim=[0.1, 1.0],
        smooth_shading=True,
        ambient=0.35,
        specular=1.0
    )

    if bars:
        pass
        #scene.add_scalar_bar(
        #    title="RWC [kg / m³]",
        #    vertical=True,
        #    position_x=0.9,
        #)

    x_l, y_l = area.get_xy_from_lonlat(lons[:, 0], lats[:, 0])
    x_r, y_r = area.get_xy_from_lonlat(lons[:, -1], lats[:, -1])

    i_start, i_end = np.where((~x_l.mask) * (~y_l.mask))[0][[0, -1]]
    x_l = x_l[i_start:i_end]
    y_l = y_l[i_start:i_end]

    i_start, i_end = np.where((~x_r.mask) * (~y_r.mask))[0][[0, -1]]
    x_r = x_r[i_start:i_end]
    y_r = y_r[i_start:i_end]

    alt = 500.0 / 1e3
    coords = np.stack([
        (xyz[::-1, :, :, 0])[y_l.data, x_l.data, 1],
        (xyz[::-1, :, :, 1])[y_l.data, x_l.data, 1],
        (xyz[::-1, :, :, 2])[y_l.data, x_l.data, 1] * 10
    ], axis=-1)
    line_l = pv.Spline(coords)
    scene.add_mesh(line_l, color="w", line_width=2)

    coords = np.stack([
        (xyz[::-1, :, :, 0])[y_r.data, x_r.data, 1],
        (xyz[::-1, :, :, 1])[y_r.data, x_r.data, 1],
        (xyz[::-1, :, :, 2])[y_r.data, x_r.data, 1] * 10
    ], axis=-1)
    line_r = pv.Spline(coords)
    scene.add_mesh(line_r, color="w", line_width=2)
