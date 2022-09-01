import time
import datetime
import pandas as pd
from dask import dataframe as dd, bag as db
import dask
from dask.distributed import Client
import numpy as np
import os
import sys
import argparse
import json
import boto3
import tempfile
import multiprocessing as mp


parser = argparse.ArgumentParser(description="Process Duo or Azure logs for DFP")
parser.add_argument('--origin', choices=['duo', 'azure'], default='duo', help='the type of logs to process: duo or azure')
parser.add_argument('--s3', action='store_true', help='Whether to load the files from s3')
parser.add_argument('--files', default=None, help='The directory or bucket containing the files to process')
parser.add_argument('--aws_key', default=None, help='The AWS Access key to use for s3 loading')
parser.add_argument('--aws_secret', default=None, help='The AWS Secret key to use for s3 loading')
parser.add_argument('--aws_token', default=None, help='The AWS Token to use for s3 loading')
parser.add_argument('--temp_save_s3', default=None, help='Save the bucket contents to a temporary directory and process them locally')
parser.add_argument('--save_dir', default=None, help='The directory to save the processed files')
parser.add_argument('--filetype', default='csv', choices=['csv', 'json', 'jsonline', 'dict'], help='Switch between csv and jsonlines for processing Azure logs')
parser.add_argument('--sep', default='_', help='The seperator between nested json keys')
parser.add_argument('--explode_raw', action='store_true', help='Option to explode the _raw key from a jsonline file')
parser.add_argument('--delimiter', default=',', help='The CSV delimiter in the files to be processed')
parser.add_argument('--groupby', default=None, help='The column to be aggregated over. Usually a username.')
parser.add_argument('--timestamp', default=None, help='The name of the column containing the timing info')
parser.add_argument('--city', default=None, help='The name of the column containing the city')
parser.add_argument('--state', default=None, help="the name of the column containing the state")
parser.add_argument('--country', default=None, help="The name of the column containing the country")
parser.add_argument('--app', default=None, help="The name of the column containing the application. Does not apply to Duo logs.")
parser.add_argument('--manager', default=None, help='The column containing the manager name. Leave blank if you want user-level results')
parser.add_argument('--extension', default=None, help='The extensions of the files to be loaded. Only needed if there are other files in the directory containing the files to be processed')
parser.add_argument('--min_records', type=int, default=0, help='The minimum number of records needed for a processed user to be saved.')


_DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'

s3_session = None
s3_client = None
s3_resource = None
def initialize_s3(aws_key, aws_secret, aws_token):
    global s3_session
    global s3_client
    global s3_resource
    s3_session = boto3.Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, aws_session_token=aws_token)
    s3_client = s3_session.client('s3')
    s3_resource = s3_session.resource('s3')

def _if_dir_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _explode_raw(df, sep):
    df2 = pd.json_normalize(df['_raw'].apply(json.loads), sep=sep)
    return df2

def _nondask_explode_raw(file, sep):
    pdf = pd.json_normalize(pd.read_json(file, lines=True)['_raw'].apply(json.loads), sep=sep)
    return pdf

def _derived_features(df, timestamp_column, city_column, state_column, country_column, application_column, normalize_strings):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column], errors='coerce')
    pdf['day'] = pdf['time'].dt.date
    pdf.fillna({'time': pd.to_datetime(_DEFAULT_DATE), 'day': pd.to_datetime(_DEFAULT_DATE).date()}, inplace = True)
    pdf.sort_values(by=['time'], inplace=True)
    overall_location_columns = [col for col in [city_column, state_column, country_column] if col is not None]
    if len(overall_location_columns) > 0:
        pdf['overall_location'] = pdf[overall_location_columns].apply(lambda x: ', '.join(x), axis=1)
        pdf['loc_cat'] = pdf.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
        pdf.fillna({'loc_cat': 1}, inplace = True)
        pdf['locincrement'] = pdf.groupby('day')['loc_cat'].expanding(1).max().droplevel(0)
        pdf.drop(['overall_location', 'loc_cat'], inplace=True, axis=1)
    if application_column is not None:
        pdf['app_cat'] = pdf.groupby('day')[application_column].transform(lambda x: pd.factorize(x)[0] + 1)
        pdf.fillna({'app_cat': 1}, inplace = True)
        pdf['appincrement'] = pdf.groupby('day')['app_cat'].expanding(1).max().droplevel(0)
        pdf.drop('app_cat', inplace=True, axis=1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    if normalize_strings:
        for feature_col in normalize_strings:
            if feature_col in pdf.columns:
                pdf[feature_col] = pdf[feature_col].str.lower()
                pdf[feature_col] = pdf[feature_col].str.replace(" ", "_")
    return pdf


def _save_groups(df, outdir, source):
    df.to_csv(os.path.join(outdir, df.name.split('@')[0]+"_"+source+".csv"), index=False)
    return df


def _parse_time(df, timestamp_column):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column])
    pdf['day'] = pdf['time'].dt.date
    return pdf


def _s3_load(bucket, key, filetype, explode_raw, delimiter, sep):
    # session = boto3.Session(aws_access_key_id=access, aws_secret_access_key=secret, aws_session_token=token)
    # client = session.client('s3')
    data = s3_client.get_object(Bucket=bucket, Key=key)
    contents = data['Body']
    if filetype.startswith('json'):
        log = json.load(contents)
        if explode_raw:
            pdf = pd.json_normalize(log['_raw'], sep=sep)
        else:
            pdf = pd.json_normalize(log, sep=sep)
    elif filetype == 'dict':
        log = json.load(contents)
        pdf = pd.DataFrame.from_dict(log)
    else:
        pdf = pd.read_csv(contents, delimiter=delimiter).fillna
    return pdf

def _load_json(file, sep):
    with open(file) as json_in:
        log = json.load(json_in)
    pdf = pd.json_normalize(log, sep=sep)
    return pdf

def _load_dict(file, sep):
    with open(file) as dict_in:
        log = json.load(dict_in)
    pdf_reoriented = pd.DataFrame.from_dict(log).to_dict('records')
    pdf = pd.json_normalize(pdf_reoriented, sep=sep)
    return pdf

def _load_csv(file, delimiter):
    return pd.read_csv(file, delimiter=delimiter)
    
def _download_s3(job):
    bucket, key, savepath = job
    s3_client.download_file(bucket, key, savepath)

def proc_logs(files, 
                save_dir,
                log_source = 'duo',
                filetype = 'csv',
                sep = '_',
                s3 = False,
                aws_key = None,
                aws_secret = None,
                aws_token = None,
                temp_save_s3 = None,
                dask = True,
                max_workers = 4,
                explode_raw = False,
                delimiter = ',',
                groupby = 'userPrincipalName',
                timestamp_column = 'createdDateTime',
                city_column = None,
                state_column = None,
                country_column = None,
                application_column = None,
                normalize_strings = None,
                output_grouping = None,
                extension=None,
                min_records = 0):
    """
    Process log files for DFP training.
    
    Parameters
    ----------
    files: str or List[str]
        A directory or filepath or list of filepaths
    save_dir: str
        The directory to save the training data
    log_source: str
        The source of the logs. Used primarily for tracing training data provenance.
    filetype: str, default='csv'
        'csv', 'json', or 'jsonline'
    sep: str, default='.'
        The character to delimit nested json keys.
    s3: bool
        Flag to indicate data should be loaded from s3
    aws_key: str
        AWS Access Key
    aws_secret: str
        AWS Secret Key
    aws_token: str
        AWS Token
    explode_raw: bool
        This indicates that the data is in a nested jsonlines object with the _raw key
    delimiter: str, default=','
        The csv delimiter
    groupby: str
        The column name to aggregate over for derived feature creation.
    timestamp_column: str, default='createdDateTime
        The column name containing the timestamp
    city_column: str
        The column name containing the city location data
    state_column: str
        The column name containing the state location data
    country_column: str
        The column name containing the country location data
    application_column: str
        The column name containing the app name data
    output_grouping: str, optional
        The column to aggregate the output training data. If None, this defaults to the aggregation level specified in the groupby parameter.
        This is where you would specify the manager name column, if training is being done by manager group.
    extension: str, optional
        Specify the file extension to load, if the directory contains additional files that should not be loaded.
    min_records: int, default=0
        The minimum number of records that need to be observed to save the data for training. Setting this to 0 creates data for all users.
    
    Returns
    -------
    bool
        True if more than 1 training file is returned, else False is returned

    """
    start_time = time.perf_counter()

    if output_grouping is None:
        output_grouping = groupby
    if isinstance(normalize_strings, str):
        normalize_strings = [normalize_strings]
    if not isinstance(normalize_strings, list):
        normalize_strings = None

    _if_dir_not_exists(save_dir)
    
    if temp_save_s3:
        _if_dir_not_exists(temp_save_s3)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_save_s3)

    if s3:
        if '/' in files:
            split_bucket = files.split('/')
            bucket = split_bucket[0]
            prefix = '/'.join(split_bucket[1:])
        else:
            bucket = files
            prefix = None
        # session = boto3.Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, aws_session_token=aws_token)
        # client = session.client('s3')
        # s3 = session.resource('s3')
        initialize_s3(aws_key, aws_secret, aws_token)
        keys = []
        if prefix is not None:
            for content in s3_resource.Bucket(bucket).objects.filter(Prefix=prefix):
                key = content.key
                keys.append(key)
        else:
            for content in s3_resource.Bucket(bucket).objects.all():
                key = content.key
                if not key.startswith('/'):
                    keys.append(key)
        if extension is not None:
            keys = [key for key in keys if key.endswith(extension)]
        assert len(keys) > 0, 'Please pass a bucket with correct prefixes containing the files to be processed'
        if temp_save_s3:
            jobs = [(bucket, k, os.path.join(temp_dir.name, k.split('/')[-1])) for k in keys]
            # for k in keys:
            #     s3_client.download_file(bucket, k, os.path.join(temp_dir.name, k.split('/')[-1]))
            pool = mp.Pool(max_workers)
            pool.map(_download_s3, jobs)
            pool.close()
            pool.join()
            files = [os.path.join(temp_dir.name, file) for file in os.listdir(temp_dir.name)]
        else:
            if dask:
                dfs = [dask.delayed(_s3_load)(bucket, k, filetype, explode_raw, delimiter, sep) for k in keys]
                ddfs = [dd.from_delayed(df) for df in dfs]
                logs = dd.concat(ddfs).fillna('nan')
            else:
                pool = mp.Pool(max_workers)
                dfs = [pool.appy(_s3_load, args=(bucket, k ,filetype, explode_raw, delimiter, sep)) for k in keys]
                logs = pd.concat(dfs).fillna('nan')
                pool.close()
                pool.join()
    if not s3 or temp_save_s3:
        if isinstance(files, str):
            if os.path.isdir(files):
                if extension is not None:
                    files = [os.path.join(files, file) for file in os.listdir(files) if file.endswith(extension)]
                else:
                    files = [os.path.join(files, file) for file in os.listdir(files)]
            elif os.path.isfile(files):
                files = [files]
            else:
                files = []
        assert isinstance(files, list) and len(files) > 0, 'Please pass a directory, a file-path, or a list of file-paths containing the files to be processed'
        if filetype == 'jsonline':
            if explode_raw:
                if dask:
                    nested_logs = dd.read_json(files, lines=True)
                    meta = pd.json_normalize(json.loads(nested_logs.head(1)['_raw'].to_list()[0]), sep=sep).iloc[:0,:].copy()
                    logs = nested_logs.map_partitions(lambda df: _explode_raw(df, sep), meta=meta).fillna('nan')
                else:
                    pool = mp.Pool(max_workers)
                    dfs = [pool.apply(_nondask_explode_raw, args=(file, sep)) for file in files]
                    logs = pd.concat(dfs).fillna('nan')
                    pool.close()
                    pool.join()
            else:
                if dask:
                    dfs = [dask.delayed(_load_json)(x, sep) for x in files]
                    # logs = dd.from_delayed(dfs, verify_meta=False)
                    ddfs = [dd.from_delayed(df) for df in dfs]
                    logs = dd.concat(ddfs).fillna('nan')
                else:
                    pool = mp.Pool(max_workers)
                    dfs = [pool.apply(_load_json, args=(file, sep)) for file in files]
                    logs = pd.concat(dfs).fillna('nan')
                    pool.close()
                    pool.join()
        elif filetype == 'json':
            if dask:
                dfs = [dask.delayed(_load_json)(x, sep) for x in files]
                # logs = dd.from_delayed(dfs, verify_meta=False)
                ddfs = [dd.from_delayed(df) for df in dfs]
                logs = dd.concat(ddfs).fillna('nan')
            else:
                pool = mp.Pool(max_workers)
                dfs = [pool.apply(_load_json, args=(file, sep)) for file in files]
                logs = pd.concat(dfs).fillna('nan')
                pool.close()
                pool.join()
        elif filetype == 'dict':
            if dask:
                dfs = [dask.delayed(_load_dict)(x, sep) for x in files]
                ddfs = [dd.from_delayed(df) for df in dfs]
                logs = dd.concat(ddfs).fillna('nan')
            else:
                pool = mp.Pool(max_workers)
                dfs = [pool.apply(_load_dict, args=(file, sep)) for file in files]
                logs = pd.concat(dfs).fillna('nan')
                pool.close()
                pool.join()
        else:
            if dask:
                logs = dd.read_csv(files, delimiter=delimiter, dtype='object').fillna('nan')
            else:
                pool.mp.Pool(max_workers)
                dfs = [pool.apply(_load_csv, args=(file, delimiter)) for file in files]
                logs = pd.concat(dfs).fillna('nan')
                pool.close()
                pool.join()

    if dask:
        logs_meta = {c: v for c, v in zip(logs._meta, logs._meta.dtypes)}
        logs_meta['time'] = 'datetime64[ns]'
        logs_meta['day'] = 'datetime64[ns]'
        if city_column is not None or state_column is not None or country_column is not None:
            logs_meta['locincrement'] = 'int'
        if application_column is not None:
            logs_meta['appincrement'] = 'int'
        logs_meta['logcount'] = 'int'

        derived_logs = logs.groupby(groupby).apply(lambda df: _derived_features(df, timestamp_column, city_column, state_column, country_column, application_column, normalize_strings), meta=logs_meta).reset_index(drop=True)

        # derived_meta = derived_logs.head(1).iloc[:0,:].copy()

        if min_records > 0:
            logs = logs.persist()
            user_entry_counts = logs[[groupby, timestamp_column]].groupby(groupby).count().compute()
            trainees = [user for user, count in user_entry_counts.to_dict()[timestamp_column].items() if count > min_records]
            derived_logs[derived_logs[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source), meta=derived_logs._meta).size.compute()
        else:
            derived_logs.groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source), meta=logs_meta).size.compute()
    else:
        derived_logs = logs.groupby(groupby).apply(lambda df: _derived_features(df, timestamp_column, city_column, state_column, country_column, application_column, normalize_strings)).reset_index(drop=True)

        if min_records > 0:
            user_entry_counts = logs[[groupby, timestamp_column]].groupby(groupby).count()
            trainees = [user for user, count in user_entry_counts.to_dict()[timestamp_column].items() if count > min_records]
            derived_logs[derived_logs[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source))
        else:
            derived_logs.groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source))
            
    if temp_save_s3:
        temp_dir.cleanup()

    timing = datetime.timedelta(seconds = time.perf_counter() - start_time)

    num_training_files = len([file for file in os.listdir(save_dir) if file.endswith('_{log_source}.csv'.format(log_source=log_source))])
    print("{num_files} training files successfully created in {time}".format(num_files=num_training_files, time=timing))
    if num_training_files > 0:
        return True
    else:
        return False


def _run():
    opt = parser.parse_args()

    if opt.dask:
        client = Client()
        client.restart()

    print('Beginning {origin} pre-processing'.format(origin=opt.origin))
    proc_logs(files=opt.files, 
                        log_source=opt.origin,
                        save_dir=opt.save_dir,
                        filetype=opt.filetype,
                        sep=opt.sep,
                        s3=opt.s3,
                        aws_key=opt.aws_key,
                        aws_secret=opt.aws_secret,
                        aws_token=opt.aws_token,
                        temp_save_s3=opt.temp_save_s3,
                        explode_raw=opt.explode_raw,
                        delimiter=opt.delimiter, 
                        groupby=opt.groupby or 'userPrincipalName',
                        timestamp_column=opt.timestamp or 'createdDateTime',
                        city_column=opt.city,
                        state_column=opt.state,
                        country_column=opt.country,
                        application_column=opt.app,
                        output_grouping=opt.manager,
                        extension=opt.extension,
                        min_records=opt.min_records)
    
    if opt.dask:
        client.close()

if __name__ == '__main__':
    _run()