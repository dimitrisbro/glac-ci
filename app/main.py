# Anomaly detection with ML algorithms for .csv input.
# Orestis Vantzos, based on code by Theodoros Grigorakakis
# Validation version, targeting a run-time of 1 hour with the default args.
#
# usage: python app.py [-h] [--in_folder IN_FOLDER] [--out_folder OUT_FOLDER] [--debug | --no-debug] [--rm | --no-rm]
#                      [--isolation_forest | --no-isolation_forest] [--contamination CONTAMINATION]
#                      [--oneclass_svm | --no-oneclass_svm] [--sample_rate SAMPLE_RATE] [--nu NU]
#
# options:
#   -h, --help            show this help message and exit
#   --in_folder IN_FOLDER
#                         [.] Input folder to scan for .csv files.
#   --out_folder OUT_FOLDER
#                         [output] Output folder to place generated .csv and .png files.
#   --debug, --no-debug   [False] Whether to raise exceptions, killing the run.
#   --rm, --no-rm         [False] Whether to remove the analysed files.
#   --isolation_forest, --no-isolation_forest
#                         [False] Whether to run the Isolation Forest algorithm
#   --contamination CONTAMINATION
#                         [0.1] Contamination param for Isolation Forest.
#   --oneclass_svm, --no-oneclass_svm
#                         [True] Whether to run the One-Class SVM algorithm.
#   --sample_rate SAMPLE_RATE
#                         [0.4] What percentage of the sample to train One-Class SVM on.
#   --nu NU               [0.2] Nu parameter for One-Class SVM.
#
# Input file format: .csv files with the index column marked 'T' with sequential int data, and at least one float-valued
# column for analysis. Output file formats: .csv with anomalies detected in a column with values 1 (normal) or -1
# (anomalous) - plot of anomalous data points in .png.

import time
import sys
from pathlib import Path
import argparse as arg
from typing import Optional

import pandas as pd
pd.options.mode.copy_on_write = True    # suppresses warnings about df[col] = ...

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print(f'Time passed: {t_hour} hr: {t_min} min: {t_sec} sec')

def output_path(in_file:str|Path, to:str, suffix:str)->Path:
    """Generate a new file path, given an output folder and a new suffix."""
    return Path(to).joinpath(Path(in_file).stem + '.' + suffix)

def csv2df(input_file:str|Path, query:Optional[str]=None)->pd.DataFrame:
    """Read an .csv into a Pandas DataFrame.
    Filters the rows."""
    print(f'Reading {input_file} ...')
    tic()
    # Read csv
    data_input = pd.read_csv(input_file, index_col='T')
    print(f'Read {len(data_input)} rows.')
    # Filter by query
    if query:
        data_input.query(query, inplace=True)
        print(f'{len(data_input)} rows after filtering.')
    tac()
    return data_input

def float_cols(df:pd.DataFrame)->list[str]:
    """Returns the names of the float-valued columns of the dataframe."""
    return df.select_dtypes([float]).columns.tolist()

def isolation_forest(data:pd.DataFrame, cols:list[str], out_col:str='Isolation_Forest', contamination:float|str='auto', n_jobs:Optional[int]=None)->pd.DataFrame:
    """Anomaly detection by Isolation Forest algo.
    Uses only the columns in cols.
    Contamination rate is 'auto' or float in (0,0.5)."""
    print(f"Running Isolation Forest Algorithm with contamination rate {contamination}...")
    anom = pd.Series(IsolationForest(random_state=0,contamination=contamination, n_jobs=n_jobs).fit_predict(data[cols]), name=out_col)
    anom_data = data.join(anom)
    anomaly_rate = anom_data[out_col].value_counts()[-1]/len(anom_data[out_col])
    print(f'Detected anomaly rate: {anomaly_rate:.1%}')
    return anom_data

def one_class_SVM(data:pd.DataFrame, cols:list[str], out_col:str='One_Class_SVM', sample_rate:float=.2, nu:float=0.01)->pd.DataFrame:
    """Anomaly detection by One Class SVM algo.
    Uses only the columns in cols.
    Randomly sample a fraction of the dataset and fit, then predict on the whol dataset.
    Higher nu, longer run-time"""
    print(f"Running One Class SVM Algorithm with nu {nu} and sample rate {sample_rate}...")
    data_fit = data[cols]
    sample = data_fit.sample(n = int(len(data)*sample_rate), random_state=42)
    anom = pd.Series(OneClassSVM(nu=nu).fit(sample).predict(data_fit), name=out_col)
    anom_data = data.join(anom)
    anomaly_rate = anom_data[out_col].value_counts()[-1]/len(anom_data[out_col])
    print(f'Detected anomaly rate: {anomaly_rate:.1%}')
    return anom_data

def plot_anomaly_data(data:pd.DataFrame, anom_col:str, val_col:Optional[str]=None, t_col:str='T')->plt.Figure:
    """Returns a fig with the detected anomalies plotted on top of the timeseries."""
    fig = plt.figure(figsize=(15,4)) 
    ax1=plt.subplot(111)
    if not val_col:
        val_col = float_cols(data)[0]    # use the first float col by default
    sns.lineplot(x=t_col, y=val_col, data=data, ax=ax1) # plot normal time series plot
    # plot subset on top of the normal time series
    scatterplot=sns.scatterplot(data=data[data[anom_col]==-1], x=t_col, y=val_col, ax=ax1, color='red',s=200)
    fig = scatterplot.get_figure()
    return fig

def get_args()->arg.Namespace:
    """Setup the argparser."""
    parser = arg.ArgumentParser(
        prog='python app.py',
        description='Anomaly detection with ML algorithms.',
        epilog="""Input file format: .csv files with the index column marked 'T' with sequential int data, and at least one float-valued column for analysis.
        Output file formats: .csv with anomalies detected in a column with values 1 (normal) or -1 (anomalous) - plot of anomalous data points in .png."""
    )

    # I/O
    parser.add_argument('--in_folder', type=str, default=".", help='[.] Input folder to scan for .csv files.')
    parser.add_argument('--out_folder', type=str, default="output", help='[output] Output folder to place generated .csv and .png files.')
    parser.add_argument('--debug', action=arg.BooleanOptionalAction, help='[False] Whether to raise exceptions, killing the run.')
    parser.set_defaults(debug=False)
    parser.add_argument('--rm', action=arg.BooleanOptionalAction, help='[False] Whether to remove the analysed files.')
    parser.set_defaults(rm=False)
    parser.add_argument('--mthread', action=arg.BooleanOptionalAction, help='[False] Use multithread algorithms where possible.')
    parser.set_defaults(mthread=False)
    # Isolation Forest
    parser.add_argument('--isolation_forest', action=arg.BooleanOptionalAction, help='[False] Whether to run the Isolation Forest algorithm')
    parser.set_defaults(isolation_forest=False)
    parser.add_argument('--contamination', type=float, default=0.1, help='[0.1] Contamination param for Isolation Forest.')
    # One-Class SVM
    parser.add_argument('--oneclass_svm', action=arg.BooleanOptionalAction, help='[True] Whether to run the One-Class SVM algorithm.')
    parser.set_defaults(oneclass_svm=True)
    parser.add_argument('--sample_rate', type=float, default=0.4, help='[0.4] What percentage of the sample to train One-Class SVM on.')
    parser.add_argument('--nu', type=float, default=0.2, help='[0.2] Nu parameter for One-Class SVM.')

    return parser.parse_args()

def post_process(anom_data:pd.DataFrame, in_file:str, out_folder:str, col:str, suffix:Optional[str]=None):
    """Save the anomaly detection results, stored in anom_data in column col.
    Output filenames based on in_file, saved in out_folder."""
    if not suffix:
        suffix=col
    # Save the annotated dataframe
    csv_file = output_path(in_file, to=out_folder, suffix=f'{suffix}.csv')
    anom_data.to_csv(csv_file, index_label='T')
    # Save plot
    png_file = output_path(in_file, to=out_folder, suffix=f'{suffix}.png')
    fig = plot_anomaly_data(anom_data, col)
    fig.savefig(png_file)

def main():
    # Check args
    args = get_args()
    print(args)
    in_folder, out_folder = args.in_folder, args.out_folder

    in_valid, out_valid = Path(in_folder).is_dir(), Path(out_folder).is_dir()
    if not (in_valid and out_valid):
        if not in_valid: print("Invalid input folder!")
        if not out_valid: print("Invalid output folder!")
        exit(-1)

    # Batch process
    batch_files = Path(in_folder).glob('*.csv')
    for data_file in batch_files:
        try:
            print("----------")
            # Import csv file
            data = csv2df(data_file)
            cols = float_cols(data)     # select all float-valued cols
            print(f'Analysing the columns: {cols}')
            
            # Detect anomalies with Isolation Forest
            if args.isolation_forest:
                tic()
                anom_data = isolation_forest(data, cols, contamination=args.contamination, out_col='Isolation_Forest', n_jobs=-1 if args.mthread else None)
                tac()
                # Save the results
                post_process(anom_data, data_file, out_folder, 'Isolation_Forest')

            # Detect anomalies with One-Class SVM
            if args.oneclass_svm:
                tic()
                anom_data = one_class_SVM(data, cols, sample_rate=args.sample_rate, nu=args.nu, out_col='One_Class_SVM')
                tac()
                # Save the results
                post_process(anom_data, data_file, out_folder, 'One_Class_SVM')
        except Exception as e:
            print(f"Processing {data_file} failed!")
            if args.debug:
                raise e
        finally:
            if args.rm:
                data_file.unlink(missing_ok=True)
                print(f"Removed {data_file}.")

if __name__=='__main__':
    main()