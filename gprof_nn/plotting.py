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
        border_color="k",
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
            edgecolor=border_color,
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
    """
    Creates a GOES 16 RGB composite for the given area.

    Args:
        time: A 'datetime' object specifying the time for which to create
            the background image.
        area: A pyresample area definition defining the area.
        filename: The file to which to store the backgroun image.
        night: If 'True', the background image will be created from
            the IR window channel.
    """
    from pansat.products.satellite.goes import GOES16L1BRadiances
    from satpy import Scene

    if not night:
        goes = GOES16L1BRadiances("F", [1, 2, 3])
        composite = "true_color"
    else:
        goes = GOES16L1BRadiances("F", [13])
        composite = "C13"
    goes_files = goes.download(time, time)

    scene = Scene([str(f) for f in goes_files], reader="abi_l1b")

    scene.load([composite])
    scene_r = scene.resample(area)
    scene_r.load([composite])
    scene_r.save_dataset(composite, filename)


def add_surface(scene, area, background, z_scaling=10):
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
    x, y, z = np.meshgrid(x, y, z)
    g = pv.StructuredGrid(
        x[:, :, :2],
        y[:, :, :2],
        z[:, :, :2] * z_scaling
    )
    g.texture_map_to_plane(
        inplace=True,
        origin=(extent[0], extent[1], 0),
        point_u=(extent[2], extent[1], 0),
        point_v=(extent[0], extent[3], 0),
    )
    texture = pv.numpy_to_texture(np.array(background)[:, :, ...,:3])
    scene.add_mesh(g, texture=texture)


def add_hydrometeors(
        scene,
        area,
        input_file,
        opacity=None,
        rwc_bar=False,
        swc_bar=False,
        z_scaling=10,
        sigma=None,
        norm=None,
        levels=None
):
    """
    Add iso-surfaces of hydrometeors to scene.

    Args:
        scene: The pyvista.Plotter containing the precipitation scene.
        area: 'pyresample' area definition defining the region of interest.
        input_file: NetCDF file containing the retrieval results to plot.
        opacity: Opacity value to use. If not given opacity will be set from
            the corresponding hydrometeor concentrations.
        rwc_bar: Whether or not to draw a scalar bar for the RWC concentrations.
        swc_bar: Whether or not to draw a scalar bar for the SWC concentrations.
        z_scaling: Scaling for the vertical dimension.
        sigma: FWHM to use for the calculation of the Gaussian weights for the
            nearest neighbor resampling.



    """
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_gauss, resample_nearest
    import pyvista as pv

    with xr.open_dataset(input_file) as data:

        data = data.transpose("scans", "pixels", "layers")

        lats = data.latitude.data
        lons = data.longitude.data
        swath = SwathDefinition(lats=lats, lons=lons)

        sp = data.surface_precip
        rwc = data.rain_water_content.data
        swc = data.snow_water_content.data


        n_layers = swc.shape[-1]
        if sigma is not None:
            rwc_r = resample_gauss(swath, rwc, area, 10e3, sigmas=[sigma] * n_layers, fill_value=np.nan)
            swc_r = resample_gauss(swath, swc, area, 10e3, sigmas=[sigma] * n_layers, fill_value=np.nan)
        else:
            rwc_r = resample_nearest(swath, rwc, area, 10e3, fill_value=np.nan)
            swc_r = resample_nearest(swath, swc, area, 10e3, fill_value=np.nan)

        extent = area.area_extent
        m, n = area.shape
        y = np.linspace(extent[0], extent[2], m)
        x = np.linspace(extent[1], extent[3], n)
        if n_layers == 28:
            z = np.concatenate([np.linspace(0, 9.5e3, 20) + 0.25e3, 10.5e3 + np.arange(0, 8) * 1e3])
        elif n_layers == 88:
            z = np.linspace(0, 22, 88 + 1)[1:] * 1e3
        else:
            z = np.linspace(0, 20, 41) * 1e3
            z = 0.5 * (z[1:] + z[:-1])
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1)


        g = pv.StructuredGrid(
            xyz[:, :, ..., 0],
            xyz[:, :, ..., 1],
            xyz[:, :, ..., 2] * z_scaling
        )

        if norm is None:
            norm = LogNorm(1e-2, 1e0)

        if levels is None:
            levels = np.linspace(0, 1, 11)[1:]

        m = ScalarMappable(norm=norm, cmap="Reds")
        cm = lambda x: m.to_rgba(x)
        g.point_data["rwc"] = rwc_r[::-1].flatten(order="F")
        colors = [to_hex(c) for c in m.to_rgba(levels)]

        if opacity is None:
            rwc_opc = norm(rwc_r[::-1].flatten(order="F"))
            g.point_data["rwc_opc"] = rwc_opc
            opacity = "rwc_opc"

        scalar_bar_args = {
            "title": "RWC [g / m³]",
            "position_x": 0.05,
            "position_y": 0.25,
            "vertical": True
        }
        contours = g.contour(scalars="rwc", isosurfaces=levels)
        scene.add_mesh(
            contours,
            cmap=colors,
            opacity=opacity,
            show_scalar_bar=rwc_bar,
            scalar_bar_args=scalar_bar_args,
            clim=[0.1, 1.0],
            smooth_shading=True,
            ambient=0.35,
            specular=1.0
        )

        g.point_data["swc"] = swc_r[::-1].flatten(order="F")

        if opacity is None:
            swc_opc = norm(swc_r[::-1].flatten(order="F"))
            g.point_data["swc_opc"] = swc_opc
            opacity = "swc_opc"

        m = ScalarMappable(norm=norm, cmap="Blues")
        cm = lambda x: m.to_rgba(x)
        colors = [to_hex(c) for c in m.to_rgba(levels)]

        scalar_bar_args = {
            "title": "SWC [g / m³]",
            "position_x": 0.90,
            "position_y": 0.25,
            "vertical": True
        }
        contours = g.contour(scalars="swc", isosurfaces=levels)
        scene.add_mesh(
            contours,
            cmap=colors,
            opacity=opacity,
            show_scalar_bar=swc_bar,
            scalar_bar_args=scalar_bar_args,
            clim=[0.1, 1.0],
            smooth_shading=True,
            ambient=0.35,
            specular=1.0
        )

        x_l, y_l = area.get_xy_from_lonlat(lons[:, 0], lats[:, 0])
        x_r, y_r = area.get_xy_from_lonlat(lons[:, -1], lats[:, -1])

        masked = np.where((~x_l.mask) * (~y_l.mask))[0]
        if len(masked) > 1:
            i_start, i_end = masked[[0, -1]] 
            x_l = x_l[i_start:i_end]
            y_l = y_l[i_start:i_end]

            x = xyz[::-1, :, :, 0][y_l.data, x_l.data, 1]
            y = xyz[::-1, :, :, 1][y_l.data, x_l.data, 1]
            z = np.ones_like(x) * 1e3 * z_scaling
            coords = np.stack([x, y, z], axis=-1)
            line_l = pv.Spline(coords)
            scene.add_mesh(line_l, color="k", line_width=2)

            z = np.ones_like(x) * 15e3 * z_scaling
            coords = np.stack([x, y, z], axis=-1)
            line_l = pv.Spline(coords)
            scene.add_mesh(line_l, color="k", line_width=2)

        masked = np.where((~x_r.mask) * (~y_r.mask))[0]
        if len(masked) > 1:
            i_start, i_end = masked[[0, -1]]
            x_r = x_r[i_start:i_end]
            y_r = y_r[i_start:i_end]
            x = xyz[::-1, :, :, 0][y_r.data, x_r.data, 1]
            y = xyz[::-1, :, :, 1][y_r.data, x_r.data, 1]
            z = np.ones_like(x) * 1e3 * z_scaling
            coords = np.stack([x, y, z], axis=-1)
            line_r = pv.Spline(coords)
            scene.add_mesh(line_r, color="k", line_width=2)

            z = np.ones_like(x) * 15e3 * z_scaling
            coords = np.stack([x, y, z], axis=-1)
            line_r = pv.Spline(coords)
            scene.add_mesh(line_r, color="k", line_width=2)

        return rwc_r, swc_r


def add_swath_edges(
        scene,
        area,
        input_file,
        z_scaling=10
):
    import pyvista as pv
    with xr.open_dataset(input_file) as data:
        lats = data.latitude.data
        lons = data.longitude.data
        x_l, y_l = area.get_xy_from_lonlat(lons[:, 0], lats[:, 0])
        x_r, y_r = area.get_xy_from_lonlat(lons[:, -1], lats[:, -1])

        i_start, i_end = np.where((~x_l.mask) * (~y_l.mask))[0][[0, -1]]
        x_l = x_l[i_start:i_end]
        y_l = y_l[i_start:i_end]

        i_start, i_end = np.where((~x_r.mask) * (~y_r.mask))[0][[0, -1]]
        x_r = x_r[i_start:i_end]
        y_r = y_r[i_start:i_end]

        extent = area.area_extent
        m, n = area.shape
        y = np.linspace(extent[0], extent[2], m)
        x = np.linspace(extent[1], extent[3], n)
        z = np.concatenate([np.linspace(0, 9.5e3, 20) + 0.25e3, 10.5e3 + np.arange(0, 8) * 1e3])
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1)

        x = xyz[::-1, :, :, 0][y_l.data, x_l.data, 1]
        y = xyz[::-1, :, :, 1][y_l.data, x_l.data, 1]
        z = np.ones_like(x) * 1e3 * z_scaling
        coords = np.stack([x, y, z], axis=-1)
        line_l = pv.Spline(coords)
        scene.add_mesh(line_l, color="k", line_width=2)

        x = xyz[::-1, :, :, 0][y_r.data, x_r.data, 1]
        y = xyz[::-1, :, :, 1][y_r.data, x_r.data, 1]
        z = np.ones_like(x) * 1e3 * z_scaling
        coords = np.stack([x, y, z], axis=-1)
        line_r = pv.Spline(coords)
        scene.add_mesh(line_r, color="k", line_width=2)
