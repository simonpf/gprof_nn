# Adding New Sensors

Adding a new senor to ``gprof_nn`` only requires adding a new senor ``*.toml`` file to the package.


## Sensor Files

The sensor files are located in ``src/gprof_nn/sensors``. There are two types of sensor files: generic sensor file describe sensor properties for a sensor type on multiple platforms. These files use the file name pattern ``<sensor_name>.toml``. Specific sensor files describe a specific sensor on a platform using the file name patter ``<sensor_name>_<platform>.toml``. The generic sensor file acts as a fallback in case no specific sensor file is found for the sensor. Specific sensor files are only required for sensors for which finetuning data is to be extracted for specific sensors.

### Basic Structure

An example of a sensor definition file is shown below. This specific file
describes the AMSR2 sensor, its viewing geometry, and the channel configuration
used by ``gprof_nn``.

The `[platform]` section identifies the satellite and instrument configuration.
It specifies the platform name, the location of the corresponding L1C input
files, and the filename prefix used to identify relevant granules.

The `[viewing_geometry]` section describes the observation geometry of the
sensor. The `kind` attribute defines the scanning technique, either `Conical` or
`CrossTrack`, while the remaining attributes specify the sensor characteristics
required to calculate its footprints on the Earth’s surface.

The `[sensor]` section defines the radiometric properties of the sensor, the
orographic-enhancement correction to apply, the data-augmentation settings, and
the PanSat products used to locate the L1C files required to generate the
fine-tuning data.

Finally, the `[sensor.gprof_channels]` table maps the sensor swaths and channels
onto the internal GPROF channel representation. Each entry assigns a pair
`(swath_index, channel_index)`, identifying the L1C swath and channel index, to
a corresponding GPROF channel slot. Note that all indices are zero-based.

```
[platform]
name = "GCOMW1"
l1c_file_path = "/pdata4/archive/GPM/1C_AMSR2_V7"
l1c_file_prefix = "1C.GCOMW1.AMSR2"

[viewing_geometry]
kind = "Conical"
altitude = 700e3
scan_range = 140.0
pixels_per_scan = 400
scan_offset = 10.2e3

[sensor]
kind = "ConstellationScanner"
frequencies = [10.65, 10.65, 18.7, 18.7, 23.8, 23.8, 36.5, 36.5, 89.0, 89.0]
offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
polarization = ["V", "H", "V", "H", "V", "H", "V", "H", "V", "H"]
orographic_enhancement = [1.14100, 1.71287, 1.46721, 1.71271, 2.96671]
earth_incidence_angle = [55.33, 55.3, 55.3, 55.28, 55.28, 55.28, 55.28, 55.3, 54.78, 54.78]
beam_width = [1.2, 1.2, 0.65, 0.65, 0.75, 0.75, 0.35, 0.35, 0.15, 0.15]
beam_width_l1c = [1.2, 1.2, 0.65, 0.65, 0.75, 0.75, 0.35, 0.35, 0.15, 0.15, 0.15, 0.15]
channel_drop = 0.1
scanline_drop = 0.1
pansat_products = ["l1c_xcal2016v_gcomw1_amsr2_v07a"]

[sensor.gprof_channels]
0 = [0, 0]
1 = [0, 1]
2 = [0, 2]
3 = [0, 3]
4 = [0, 4]
5 = [0, 5]
6 = [0, 6]
7 = [0, 7]
8 = [0, 8]
9 = [0, 9]
```

