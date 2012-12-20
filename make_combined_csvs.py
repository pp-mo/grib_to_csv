"""
  For now : produce full-area for 5 hours from a particular day.
  Note that the data arrangements in the original data is peculiar...
    the forecast periods (and times) are not regular, even for the same (phenomenon) time next day
      -- and this is in the filenames, so those aren't regular either.
      -- you must use glob, and then try to pick matching forecast periods ??
"""
import csv

import math
import datetime

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt

#details of test-data set
testcode_basepath = '~/git/grib_to_csv/testdata/'
testdata_filenames = ["201103200900_u1096_ng_ek07_wind0360_speed10m.grib", "201103200900_u1096_ng_ek07_wind0360_dir10m.grib"]


# set maximum clip region to avoid some missing values in the test data
valid_lon_lims = [-10.4, 4.0] 
valid_lat_lims = [48.3, 60.25]

def produce_csv_files(
    region=None,
    in_filenames=None, out_filepath=None,
    show_plot=False, use_subplot=None
    ):
    """
    Produce output CSV files for some data of windspeed and direction.

    region        - required output (lon-min, lat-min, lon-max, lat-max)  [default: roughly 'SW-England']
    in_filenames  - filenames of (speed,angle) input files [default: testdata examples]
    out_filepath  - pathname of output file [default: "./windspeed_direction.csv"] 
    show_plot     - pop up window with windspeed data [default: False]
    """

    # fixup path defaults
    if region is None:
        region = [(-4.5, 50.0), (-2.0, 52.5)]
    region = np.array(region).flat
    if in_filenames is None:
        in_filenames = testdata_filenames
        in_filenames = [testdata_basepath+f for f in in_filenames]
    if out_filepath is None:
        out_filepath = "./windspeed_direction.csv"
        
    # load the input data
    speed_path, angle_path = in_filenames
    speeds_full = iris.load_cube(speed_path)
    angles_full = iris.load_cube(angle_path)
    
    # set required region
    reqd_lon_lims = [region[0], region[2]] #[-4.5,-2.0]
    reqd_lat_lims = [region[1], region[3]] #[50.0, 52.5]
    
    # cutout required data block
    region_lon_lims = [max(reqd_lon_lims[0], valid_lon_lims[0]), min(reqd_lon_lims[1], valid_lon_lims[1])] 
    region_lat_lims = [max(reqd_lat_lims[0], valid_lat_lims[0]), min(reqd_lat_lims[1], valid_lat_lims[1])] 
    region_ll = iris.Constraint(
        longitude=lambda x: region_lon_lims[0] <= x <= region_lon_lims[1], 
        latitude=lambda y: region_lat_lims[0] <= y <= region_lat_lims[1]
        )
    speeds_region = speeds_full.extract(region_ll)
    angles_region = angles_full.extract(region_ll)

    # calculate     
    lats = speeds_region.coord('latitude').points
    lons = speeds_region.coord('longitude').points
    
    n_lons = len(lons)
    lon0, lon1 = lons[[0,-1]]
    d_lons = lons[1]-lons[0]
    
    n_lats = len(lats)
    lat0, lat1 = lats[[0,-1]]
    d_lats = lats[1]-lats[0]
    
#    show_plot = True
    if show_plot:
#        if use_subplot is not None:
#            plt.subplot(use_subplot)
        qplt.contourf(speeds_region, np.arange(0,25,2))
#        plt.gca().coastlines(resolution='50m')
        plt.gca().coastlines()
        # add wind arrows
        n_subsam = 25
        u_spd = speeds_region.units
        q_lons = lons[::n_subsam]
        q_lats = lats[::n_subsam]
        q_spds = u_spd.convert(speeds_region.data[::n_subsam,::n_subsam], 'knot')
        print 'max m/s = ',np.max(speeds_region.data)
        print 'max kts = ',np.max(q_spds)
        q_angs_c = angles_region.data[::n_subsam,::n_subsam] * math.pi/180.0
        arrow_speed_scaling = 0.1
        q_dxs = arrow_speed_scaling * q_spds * np.cos(q_angs_c)
        q_dys = arrow_speed_scaling * q_spds * np.sin(q_angs_c)
        plt.quiver(q_lons,q_lats,q_dxs,q_dys)
#        plt.show()
    
    #fix basic comment header
    header_lines = \
    """
    # header: latlon data grid of values
    # 
    # content:       "windspeed", "direction"
    # units:         "kt", "deg"
    # file format:
    #   n-comment-lines:            (n)
    #   comment lines [I.E. _these_ lines] ...
    #   header lines:
    #       latitude grid info:     n-lats, lat-min, lat-max, lat-step
    #       longitude grid info:    n-lons, lon-min, lon-max, lon-step
    #   data lines:
    #       n-lats * (line: n-lons * (speed, direction))
    """
    
    # convert header : split at newlines, skip blank lines, save each as a 1-element list for an output 'row'
    # split comment into lines
    header_lines = header_lines.split('\n')
    # strip out spaces
    header_lines = [l.strip() for l in header_lines]
    # each nonblank becomes a single output item (one "column" in the file)
    header_lines = [[l] for l in header_lines if len(l)]  

    array_speeds_knots = speeds_region.units.convert(speeds_region.data, 'knots')
    array_angles_degrees = angles_region.data   #angles_region.units.convert(angles_region.data, 'degrees')
    
    reshaped_data = (array_speeds_knots, array_angles_degrees)
    reshaped_data = [array.reshape([n_lats, n_lons, 1]) for array in reshaped_data]
    reshaped_data = np.concatenate(reshaped_data, 2).reshape((n_lats, 2*n_lons))
    
    with open(out_filepath, 'wb') as f:
        wr =csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        
        wr.writerow([len(header_lines)])
        wr.writerows(header_lines)
        wr.writerow([n_lats, lat0, lat1, d_lats])
        wr.writerow([n_lons, lon0, lon1, d_lons])
        wr.writerows(reshaped_data)
        
        print 'output file produced : ', out_filepath
    
    print 'Done.'


# own backup stored (for one day)
#default_data_basepath = '/data/local/itpp/Remedy/_i_csv_winds_WO0000000037553/captured_testdata/day_0801/'
#default_data_basepath = '/data/local/itpp/Remedy/_i_csv_winds_WO0000000037553/captured_testdata/day_0924/'
# public all-data store (bit massive!) 
#default_data_basepath = '/data/nwp1/cfst/MOGUK/grib/'

default_data_basepath = '~/git/grib_to_csv/testdata/'
dayhours_outpath = '~/git/grib_to_csv/testdata/output'

def pairof_MOGUK_search_filepaths(date_string, n_hour, data_basepath=default_data_basepath):
    """ Construct a pair of search strings for speed,direction files. """
    hr_string = "%02d" % n_hour
    glob_spec_strs = [data_basepath + date_string + hr_string + t_s for t_s in ('*speed10mmean.grib','*dir10mmean.grib')]
    return glob_spec_strs

import glob 
def do_day_hours(day_date_string, hour_numbers, basepath=None, show_plots=False):
    n_plots = len(hour_numbers)
    n_plotrows = 2 if n_plots>3 else 1 # for now
    if show_plots:
#        plt.interactive(True)
        plt.figure()
    for (i_plot, hr) in enumerate(hour_numbers):
        # get 2 filespecs to search for datafiles
        spec_strings = pairof_MOGUK_search_filepaths(day_date_string, hr)
        # for now, take **alphabetical last** of files matching target time (==latest forecast date)
        print "spec_strings", spec_strings
        filepair = [sorted(glob.glob(spec))[-1] for spec in spec_strings]
        base_outname = "alluk_%6s_%02d_" % (day_date_string, hr)
        outdir_path=dayhours_outpath+('day_%s/' % day_date_string[-4:])
        out_filepath=outdir_path+base_outname+'spd_and_dir.csv'
        
        if show_plots:
            n_subplot = 100*n_plotrows + 10*((n_plots+n_plotrows-1) // n_plotrows) + i_plot+1 
            print 'subplot : ', n_subplot 
            plt.subplot(n_subplot)
        produce_csv_files(
            region=[-1000,-1000,1000,1000], 
            in_filenames=filepair, 
            out_filepath=out_filepath,
            show_plot=show_plots
        )
    if show_plots:
        plt.show()

if __name__ == '__main__':
    # script code
    
    # stored example day    
    t = 1
#      do_day_hours('20120801', [12,13,14,15,16,17])

    do_day_hours('20120924', [7,9,11,13,15,17], show_plots=False)
#    do_day_hours('20120925', [5,9,13,17,21], show_plots=True)
#    do_day_hours('20120801', [9,15,21], show_plots=True)
