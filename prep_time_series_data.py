"""
Sales Time Series Dataset Processing and Database Load
Create features at different levels of analysis and upload to PostgreSQL on AWS RDS.

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # Perform initial cleaning on train data
    python3 prep_time_series_data.py clean --to_sql

    # Create DF of shop/item/date (or combination)-level features and write to SQL table
    python3 prep_time_series_data.py shops --to_sql

    # Do a test run on last month of available data
    python3 prep_time_series_data.py dates --to_sql --test_run
"""

# Standard library imports
import argparse
import datetime
import gc
import json
import logging
import os
from pathlib import Path
import platform
import warnings

# Third-party library imports
import boto3
from botocore.exceptions import ClientError
from ec2_metadata import ec2_metadata
import numpy as np
import pandas as pd
from psycopg2.sql import SQL, Identifier
from scipy.stats import median_absolute_deviation, variation
import sqlalchemy

# Local imports
from dateconstants import (
    FIRST_DAY_OF_TRAIN_PRD,
    LAST_DAY_OF_TRAIN_PRD,
    FIRST_DAY_OF_TEST_PRD,
    PUBLIC_HOLIDAYS,
    PUBLIC_HOLIDAY_DTS,
    OLYMPICS2014,
    WORLDCUP2014,
    CITY_POP,
)
from rds_db_commands import (
    create_db_table_from_query,
    df_from_sql_table,
    df_from_sql_query,
    drop_tables,
)
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer
from write_df_to_sql_table import psql_insert_copy, write_df_to_sql

warnings.filterwarnings("ignore")

# Load data
data_path = "./Data/competitive-data-science-predict-future-sales/"

# Original sales data at roughly shop-item-date level for the train period
sales_df = pd.read_csv(data_path + "sales_train.csv")
# Original shop-item level data for the test period
test_df = pd.read_csv(data_path + "test.csv")
# Original dataset with item_id to item_category_id mapping
items_df = pd.read_csv(data_path + "items.csv")
# Original dataset with item_category_id to item_category_name mapping
categories_df = pd.read_csv(data_path + "item_categories.csv")
# Original dataset with shop_id to shop_name and city mapping
shops = pd.read_csv(data_path + "shops.csv")
# Original dataset with columns containing daily macroeconomic indicators
macro_df = pd.read_csv(data_path + "macro.csv", parse_dates=["timestamp"])
# Original dataset with PS4 game release dates
usecols = [
    "Title",
    "Genre",
    "Developer",
    "Publisher",
    "Release date JP",
    "Release date EU",
    "Release date NA",
    "Addons",
]
ps4games = pd.read_csv(data_path + "ps4_games.csv", usecols=usecols)
ps4games[["Release date JP", "Release date EU", "Release date NA"]] = ps4games[
    ["Release date JP", "Release date EU", "Release date NA"]
].apply(pd.to_datetime, errors="coerce")


def upload_file(file_name, bucket, object_name):
    """Upload a file to an S3 bucket.

    Parameters:
    -----------
    file_name : str
        File to upload
    bucket : str
        Bucket to upload to
    object_name : str
        S3 object name. If not specified then file_name is used

    Returns:
    --------
    True if file was uploaded, else False
    """

    s3 = boto3.resource("s3")
    try:
        response = s3.meta.client.upload_file(file_name, bucket, object_name)
    except ClientError:
        logging.exception("Exception occurred")


# Downcast Numeric Columns to Reduce Memory Usage
# from https://hackersandslackers.com/downcast-numerical-columns-python-pandas/
# with exclude=['uint32'] added to _downcast_all function to avoid errors on existing unsigned integer columns


def _float_to_int(ser):
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


def _multi_assign(df, transform_fn, condition):
    return df.assign(**{col: transform_fn(df[col]) for col in condition(df)})


def _all_float_to_int(df):
    transform_fn = _float_to_int
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)

    return _multi_assign(df, transform_fn, condition)


def _downcast_all(df, target_type, initial_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if initial_type is None:
        initial_type = target_type

    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)

    condition = lambda x: list(
        x.select_dtypes(include=[initial_type], exclude=["uint32"]).columns
    )

    return _multi_assign(df, transform_fn, condition)


@Timer(logger=logging.info)
def _downcast(df_in):
    return (
        df_in.pipe(_all_float_to_int)
        .pipe(_downcast_all, "float")
        .pipe(_downcast_all, "integer")
        .pipe(_downcast_all, target_type="unsigned", initial_type="integer")
    )


# based on https://stackoverflow.com/a/34384664/9987623
def _map_to_sql_dtypes(df):
    """Create dictionary mapping pandas/numpy data types in dataframe to
    PostgreSQL data types.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe with columns that need their data types mapped

    Returns:
    --------
    dtypedict
        Dictionary with keys being column names and values being PostgreSQL
        data types
    """

    dtypedict = {}
    for i, j in zip(df.columns, df.dtypes):
        if "object" in str(j):
            dtypedict.update({i: sqlalchemy.types.VARCHAR(length=255)})

        elif "datetime" in str(j):
            dtypedict.update({i: sqlalchemy.types.DateTime()})

        # PostgreSQL accepts float(1) to float(24) as selecting the real type
        # per https://www.postgresql.org/docs/12/datatype-numeric.html
        elif "float" in str(j):
            dtypedict.update({i: sqlalchemy.types.Float(precision=3, asdecimal=True)})

        elif ("int64" in str(j)) | ("uint32" in str(j)):
            dtypedict.update({i: sqlalchemy.types.BIGINT()})

        elif ("int32" in str(j)) | ("uint16" in str(j)):
            dtypedict.update({i: sqlalchemy.types.INT()})

        elif ("int8" in str(j)) | ("int16" in str(j)):
            dtypedict.update({i: sqlalchemy.types.SMALLINT()})

    return dtypedict


# PERFORM INITIAL DATA CLEANING
@Timer(logger=logging.info)
def clean_sales_data(return_df=False, to_sql=False, test_run=False):
    """Perform initial data cleaning of the train shop-item-date data.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    sales : pandas dataframe
        Cleaned-up version of shop-item-date train dataset
    """

    sales = sales_df.copy()

    # convert the date column from string to datetime type
    sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y"))

    # Drop duplicate rows
    sales.drop_duplicates(inplace=True)

    # Identify duplicate rows by shop-item-date
    dupes = sales.loc[
        sales.duplicated(subset=["shop_id", "item_id", "date"], keep=False), :
    ]

    # Identify shop-item-date combinations to remove because of multiple quantities
    # where one quantity is negative
    to_remove = (
        dupes.groupby(["shop_id", "item_id", "date"])
        .item_cnt_day.apply(lambda x: ((x > 0).any()) & ((x < 0).any()))
        .reset_index(name="to_remove")
        .query("to_remove == True")
        .drop("to_remove", axis=1)
    )

    # Remove those combinations from the dupes dataframe
    dupes = (
        pd.merge(dupes, to_remove, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )

    # combine remaining shop-item-date-price level values into shop-item-date level values
    # by summing the quantity sold and taking the weighted average of price (weighted by quantity)

    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=dupes.loc[x.index, "item_cnt_day"])

    dupes = (
        dupes.groupby(["shop_id", "item_id", "date", "date_block_num"])
        .agg({"item_cnt_day": "sum", "item_price": wm})
        .reset_index()
    )

    # remove the manipulated rows from the original dataframe
    sales.drop_duplicates(
        subset=["shop_id", "item_id", "date"], keep=False, inplace=True
    )

    # insert the new version of those rows back into the original dataframe
    sales = pd.concat([sales, dupes], axis=0, sort=True).reset_index(drop=True)

    # remove row with negative price
    sales.query("item_price > 0.0", inplace=True)

    # sort the dataset
    sales.sort_values(
        by=["shop_id", "item_id", "date"], inplace=True, ignore_index=True
    )

    # save DF to file
    if test_run:
        output_csv = "sales_cleaned_test_run.csv"
    else:
        output_csv = "sales_cleaned.csv"
    sales = sales[sales.date >= datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD)]
    sales.to_csv(output_csv, index=False)

    sales = _downcast(sales)
    logging.info(
        f"Sales dataframe has {sales.shape[0]} rows and " f"{sales.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Sales dataframe has the following columns: "
        f"{nl}{nl.join(sales.columns.to_list())}"
    )
    logging.info(
        f"Sales dataframe has the following data types: "
        f"{nl}{sales.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = sales.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Sales dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    # Initialize master dictionary of data types of columns from DFs that are going
    # to be uploaded to SQL tables and queried later
    # or, if already exists in JSON file, load from JSON
    input_json = "master_pd_types.json"
    types_json = Path(input_json)
    if types_json.is_file():
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)
    else:
        master_pd_types = dict()

    sales.rename(columns={"date": "sale_date"}, inplace=True)
    master_pd_types.update({"sales": sales.dtypes.map(str).to_dict()})
    with open("master_pd_types.json", "w") as fp:
        json.dump(master_pd_types, fp)

    if to_sql:
        start_instance()
        dtypes_dict = _map_to_sql_dtypes(sales)
        write_df_to_sql(sales, "sales_cleaned", dtypes_dict)

    if return_df:
        sales.rename(columns={"sale_date": "date"}, inplace=True)
        return sales


@Timer(logger=logging.info)
def _add_col_prefix(df, prefix, cols_not_to_rename=["shop_id", "item_id", "date"]):
    """Rename columns in DF for easy identification by adding a prefix to column names.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe with columns to be renamed
    prefix : str
        Prefix to prepend to column names
    cols_not_to_rename : list
        list of columns not be renamed

    Returns:
    --------
    df : pandas DataFrame
        Dataframe with renamed columns
    """

    df.columns = [
        prefix + col
        if ((col not in cols_not_to_rename) & (~col.startswith(prefix)))
        else col
        for col in df.columns
    ]
    return df


# SHOP-LEVEL FEATURES
def _lat_lon_to_float(in_coord, degree_sign="\N{DEGREE SIGN}"):
    """Convert latitude-longitude text string into latitude and longitude floats.

    Parameters:
    -----------
    in_coord : str
        latitude-longitude string (example format: '55°53′21″ с. ш. 37°26′42″ в. д.')
    degree_sign : str
        Unicode representation of degree sign

    Returns:
    --------
    geo_lat, geo_lon: float values of latitude and longitude (i.e., with minutes
        and seconds converted to decimals)
    """
    remove = degree_sign + "′" + "″"
    geo_list = (
        in_coord.translate({ord(char): " " for char in remove})
        .replace("с. ш.", ",")
        .replace("в. д.", "")
        .split(",")
    )
    if len(geo_list[0].split()) == 3:
        geo_lat = (
            float(geo_list[0].split()[0])
            + float(geo_list[0].split()[1]) / 60.0
            + float(geo_list[0].split()[2]) / 3600.0
        )
    elif len(geo_list[0].split()) == 2:
        geo_lat = float(geo_list[0].split()[0]) + float(geo_list[0].split()[1]) / 60.0
    if len(geo_list[1].split()) == 3:
        geo_lon = (
            float(geo_list[1].split()[0])
            + float(geo_list[1].split()[1]) / 60.0
            + float(geo_list[1].split()[2]) / 3600.0
        )
    elif len(geo_list[1].split()) == 2:
        geo_lon = float(geo_list[1].split()[0]) + float(geo_list[1].split()[1]) / 60.0
    return geo_lat, geo_lon


@Timer(logger=logging.info)
def build_shop_lvl_features(return_df=False, to_sql=False):
    """Build dataframe of shop-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    shops_df : pandas dataframe
        Dataframe with each row representing a shop and columns containing shop-level features
    """

    # remove shop_ids 9 and 20 (as they were found to have strange sales trends)
    shops_df = shops[~(shops.shop_id.isin([9, 20]))]

    # Shop City

    # create city of shop column
    shops_df.loc[0, "shop_name"] = "Якутск Орджоникидзе, 56 фран"
    shops_df.loc[1, "shop_name"] = 'Якутск ТЦ "Центральный" фран'

    shops_df["city"] = shops_df.shop_name.apply(lambda x: x.split()[0])

    shops_df.loc[55, "city"] = "Интернет-магазин"

    # City Population, Latitude and Longitude, and Time Zone

    city_df = pd.DataFrame(
        CITY_POP, columns=["city", "population", "geo_coords", "time_zone"]
    )

    all_lat_lons = city_df.geo_coords.apply(_lat_lon_to_float)

    city_df["geo_lat"] = all_lat_lons.apply(lambda x: x[0])
    city_df["geo_lon"] = all_lat_lons.apply(lambda x: x[1])

    city_df.drop(columns="geo_coords", inplace=True)

    city_df["time_zone"] = city_df.time_zone.apply(lambda x: x[-1]).astype(np.int8)

    shops_df = shops_df.merge(city_df, on="city", how="left")

    # Indicator Column for Online Store

    # create indicator column for online store
    shops_df["online_store"] = np.where(shops_df.city == "Интернет-магазин", 1, 0)

    # Count of Other Shops in Same City

    # create column with count of other shops in same city
    shops_df["n_other_stores_in_city"] = (
        shops_df.groupby("city").city.transform("count") - 1
    )

    shops_df.loc[
        shops_df[shops_df.city == "Интернет-магазин"].index, ["n_other_stores_in_city"]
    ] = np.nan

    # assign values of physical features of Moscow-based shops to the two online stores
    moscow_shop_features = (
        shops_df.query("city == 'Москва'")
        .head(1)
        .drop(["shop_name", "city", "shop_id", "online_store"], axis=1)
        .to_dict(orient="records")[0]
    )
    shops_df.fillna(moscow_shop_features, inplace=True)

    shops_df = _downcast(shops_df)
    shops_df = _add_col_prefix(shops_df, "s_")

    logging.info(
        f"Shops dataframe has {shops_df.shape[0]} rows and "
        f"{shops_df.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Shops dataframe has the following columns: "
        f"{nl}{nl.join(shops_df.columns.to_list())}"
    )
    logging.info(
        f"Shops dataframe has the following data types: "
        f"{nl}{shops_df.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = shops_df.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Shops dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    if to_sql:
        start_instance()
        dtypes_dict = _map_to_sql_dtypes(shops_df)
        write_df_to_sql(shops_df, "shops", dtypes_dict)

    if return_df:
        return shops_df


# ITEM-LEVEL FEATURES
def _group_game_consoles(cat_name):
    """Create modified category name column where items related to same kind of game console are grouped.

    Parameters:
    -----------
    cat_name : str
        Original category name

    Returns:
    --------
    modified category name
    """

    if "PS2" in cat_name:
        return "PS2"
    elif "PS3" in cat_name:
        return "PS3"
    elif "PS4" in cat_name:
        return "PS4"
    elif "PSP" in cat_name:
        return "PSP"
    elif "PSVita" in cat_name:
        return "PSVita"
    elif "XBOX 360" in cat_name:
        return "XBOX 360"
    elif "XBOX ONE" in cat_name:
        return "XBOX ONE"
    elif "Игры PC" in cat_name:
        return "Игры PC"
    return cat_name


@Timer(logger=logging.info)
def build_item_lvl_features(return_df=False, to_sql=False, test_run=False):
    """Build dataframe of item-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    item_level_features : pandas dataframe
        Dataframe with each row representing an item and columns containing item-level features
    """
    # check if cleaned sales file already exists
    if test_run:
        input_csv = "sales_cleaned_test_run.csv"
    else:
        input_csv = "sales_cleaned.csv"
    cleaned_sales_file = Path(input_csv)
    if cleaned_sales_file.is_file():
        sales = pd.read_csv(input_csv)
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(return_df=True, test_run=test_run)

    # Column of all unique item_ids
    item_level_features = (
        sales[["item_id"]]
        .drop_duplicates()
        .sort_values(by="item_id")
        .reset_index(drop=True)
    )

    # Item Name, Category ID, Category Name

    # Add item_name, item_category_id, item_category_name columns
    item_level_features = item_level_features.merge(items_df, on="item_id", how="left")
    item_level_features = item_level_features.merge(
        categories_df, on="item_category_id", how="left"
    )

    # Other Versions of Item Category

    # Create broad category name column
    item_level_features[
        "item_category_broad"
    ] = item_level_features.item_category_name.apply(lambda x: x.split()[0])

    item_level_features[
        "item_cat_grouped_by_game_console"
    ] = item_level_features.item_category_name.apply(_group_game_consoles)

    # Indicator for Digital Item

    # Create indicator column for whether item is digital
    item_level_features["digital_item"] = np.where(
        (item_level_features.item_category_name.str.contains("Цифра"))
        | (item_level_features.item_category_name.str.contains("MP3")),
        1,
        0,
    )

    # Indicator of Month of Year When Item Was First Sold

    item_level_features["item_mon_of_first_sale"] = (
        sales.groupby("item_id")["date"].min().dt.month.values
    )

    # Replace 1's with 0's for items that show first date of sale in Jan 2013

    jan13 = (
        (sales.groupby("item_id")["date"].min().dt.year == 2013)
        & (sales.groupby("item_id")["date"].min().dt.month == 1)
    ).values

    item_level_features.loc[jan13, "item_mon_of_first_sale"] = 0
    item_level_features[
        "item_mon_of_first_sale"
    ] = item_level_features.item_mon_of_first_sale.astype("category")

    item_level_features = _downcast(item_level_features)
    item_level_features = _add_col_prefix(item_level_features, "i_")

    logging.info(
        f"Item-level dataframe has {item_level_features.shape[0]} rows and "
        f"{item_level_features.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Item-level dataframe has the following columns: "
        f"{nl}{nl.join(item_level_features.columns.to_list())}"
    )
    logging.info(
        f"Item-level dataframe has the following data types: "
        f"{nl}{item_level_features.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = item_level_features.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Item-level dataframe dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    # Initialize master dictionary of data types of columns from DFs that are going
    # to be uploaded to SQL tables and queried later
    # or, if already exists in JSON file, load from JSON
    input_json = "master_pd_types.json"
    types_json = Path(input_json)
    if types_json.is_file():
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)
    else:
        master_pd_types = dict()

    master_pd_types.update({"items": item_level_features.dtypes.map(str).to_dict()})
    with open("master_pd_types.json", "w") as fp:
        json.dump(master_pd_types, fp)

    if to_sql:
        start_instance()
        dtypes_dict = _map_to_sql_dtypes(item_level_features)
        write_df_to_sql(item_level_features, "items", dtypes_dict)

    if return_df:
        return item_level_features


# DATE-LEVEL FEATURES


def _days_to_holiday(curr_dt, list_of_holidays=PUBLIC_HOLIDAY_DTS):
    """Calculate number of days left until next holiday.

    Parameters:
    -----------
    curr_dt : datetime
        Date for which days until next holiday is to be calculated
    list_of_holidays: list
        List of holiday dates

    Returns:
    --------
    n_days_left : int
        Number of days left until next holiday
    """
    n_days_left = (
        min(hol for hol in list_of_holidays if hol >= curr_dt) - curr_dt.date()
    ).days
    return n_days_left


def _days_after_holiday(curr_dt, list_of_holidays=PUBLIC_HOLIDAY_DTS):
    """Calculate number of days elapsed since most recent holiday.

    Parameters:
    -----------
    curr_dt : datetime
        Date for which days elapsed since most recent holiday is to be calculated
    list_of_holidays: list
        List of holiday dates

    Returns:
    --------
    n_days_since : int
        Number of days elapsed since most recent holiday
    """
    n_days_since = (
        curr_dt.date() - max(hol for hol in list_of_holidays if hol <= curr_dt)
    ).days
    return n_days_since


def _month_counter(d1):
    """Calculate number of months between first month of train period and specified date.

    Parameters:
    -----------
    d1 : datetime
        Date for which mount counter value is to be calculated

    Returns:
    --------
    Value of number of months between first month of train period and date passed to the function
    """
    return (
        (d1.year - datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD).year) * 12
        + d1.month
        - datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD).month
    )


@Timer(logger=logging.info)
def build_date_lvl_features(return_df=False, to_sql=False, test_run=False):
    """Build dataframe of date-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    date_level_features : pandas dataframe
        Dataframe with each row representing a date and columns containing date-level features
    """
    # check if cleaned sales file already exists
    if test_run:
        input_csv = "sales_cleaned_test_run.csv"
    else:
        input_csv = "sales_cleaned.csv"
    cleaned_sales_file = Path(input_csv)
    if cleaned_sales_file.is_file():
        sales = pd.read_csv(input_csv)
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(return_df=True, test_run=test_run)

    # Dates from Start to End of Training Period
    date_level_features = pd.DataFrame(
        {
            "date": pd.date_range(
                datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD),
                datetime.datetime(*LAST_DAY_OF_TRAIN_PRD),
            )
        }
    )

    # create same date_block_num column that exists in sales train dataset
    date_level_features["date_block_num"] = date_level_features.date.apply(
        _month_counter
    )

    # Date Counter (Linear Trend)
    date_level_features["date_counter"] = date_level_features.index + 1

    # Year of Date

    # create year column
    date_level_features["year"] = date_level_features.date.dt.year.astype("category")

    # Month of Year (0-11)

    # create month column
    date_level_features["month"] = date_level_features.date_block_num % 12

    # Days in Month

    # create days in month column
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]).to_dict()
    date_level_features["days_in_mon"] = (
        date_level_features["month"].map(days).astype(np.int8)
    )

    # Sine and Cosine of Month

    # create sine and cosine of month columns
    date_level_features["month_sin"] = np.sin(
        (date_level_features.month) * (2.0 * np.pi / 12)
    )
    date_level_features["month_cos"] = np.cos(
        (date_level_features.month) * (2.0 * np.pi / 12)
    )

    # Day of Week

    # create day of week column
    date_level_features["day_of_week"] = date_level_features.date.dt.weekday

    # Sine and Cosine of Day of Week

    # create sine and cosine of day of week columns
    date_level_features["dow_sin"] = np.sin(
        (date_level_features.day_of_week) * (2.0 * np.pi / 7)
    )
    date_level_features["dow_cos"] = np.cos(
        (date_level_features.day_of_week) * (2.0 * np.pi / 7)
    )

    # Indicator for Weekend Day

    # create indicator column for weekend days
    date_level_features["is_weekend"] = np.where(
        date_level_features.day_of_week < 5.0, 0, 1
    )

    # Quarter of Year

    # create quarter of year column
    date_level_features["quarter_of_year"] = date_level_features.date.dt.quarter

    # Sine and Cosine of Quarter of Year

    # create sine and cosine of quarter of year columns
    date_level_features["qoy_sin"] = np.sin(
        (date_level_features.quarter_of_year) * (2.0 * np.pi / 4)
    )
    date_level_features["qoy_cos"] = np.cos(
        (date_level_features.quarter_of_year) * (2.0 * np.pi / 4)
    )

    # Continuous Quarter of Year

    # create continuous quarter of year column
    date_level_features["quarter_counter"] = (
        date_level_features.date_block_num // 3
    ) + 1

    # Week of Year

    # create week of year column
    date_level_features["week_of_year"] = date_level_features.date.dt.isocalendar().week

    # Sine and Cosine of Week of Year

    # create sine and cosine of week of year columns
    date_level_features["woy_sin"] = np.sin(
        (date_level_features.week_of_year) * (2.0 * np.pi / 52)
    )
    date_level_features["woy_cos"] = np.cos(
        (date_level_features.week_of_year) * (2.0 * np.pi / 52)
    )

    # Indicator for Public Holiday

    # create indicator column for whether date is a public holiday
    date_level_features["holiday"] = date_level_features.date.isin(
        PUBLIC_HOLIDAYS
    ).astype(np.int8)

    # Number of Days Before a Holiday and Number of Days After a Holiday

    date_level_features["days_to_holiday"] = date_level_features["date"].apply(
        _days_to_holiday
    )
    date_level_features["days_after_holiday"] = date_level_features["date"].apply(
        _days_after_holiday
    )

    # Indicator for Major Event

    # create indicator column for major events
    olympics = pd.date_range(*OLYMPICS2014).to_series().values
    world_cup = pd.date_range(*WORLDCUP2014).to_series().values
    major_events = np.concatenate([olympics, world_cup])
    date_level_features["major_event"] = date_level_features.date.isin(
        major_events
    ).astype(np.int8)

    # Macroeconomic Indicator Columns

    # subset macro_df dataset to relevant period
    macro_df_2013_2015 = macro_df.rename(columns={"timestamp": "date"})
    macro_df_2013_2015 = macro_df_2013_2015[
        (macro_df_2013_2015.date >= datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD))
        & (macro_df_2013_2015.date <= datetime.datetime(*LAST_DAY_OF_TRAIN_PRD))
    ]

    # identify columns in macro_df dataset that have no null values
    macro_nulls = (
        macro_df_2013_2015.isnull()
        .sum(axis=0)
        .reset_index()
        .rename(columns={"index": "column", 0: "count"})
    )
    cols_wo_nulls = np.array(macro_nulls[macro_nulls["count"] == 0]["column"])

    # Frequency of update of each of these columns:
    #
    # **daily**: brent, usdrub, eurrub, rts, micex, micex_cbi_tr, micex_rgbi_tr
    # **monthly**: oil_urals, cpi, ppi, balance_trade, balance_trade_growth (only 12 unique), deposits_value, deposits_growth, deposits_rate, mortgage_value, mortgage_growth, mortgage_rate, income_per_cap, fixed_basket, rent_price_4+room_bus, rent_price_3room_bus, rent_price_2room_bus, rent_price_1room_bus, rent_price_3room_eco, rent_price_2room_eco, rent_price_1room_eco
    # **quarterly**: average_provision_of_build_contract, average_provision_of_build_contract_moscow, gdp_quart, gdp_quart_growth
    # **annual**: gdp_deflator, gdp_annual, gdp_annual_growth, salary, salary_growth, retail_trade_turnover, retail_trade_turnover_growth, retail_trade_turnover_per_cap, labor_force, unemployment, employment, invest_fixed_capital_per_cap, invest_fixed_assets, pop_natural_increase, childbirth, mortality, average_life_exp, load_of_teachers_school_per_teacher, students_state_oneshift, modern_education_share, old_education_build_share, provision_nurse, load_on_doctors, turnover_catering_per_cap, seats_theather_rfmin_per_100000_cap, bandwidth_sports, apartment_fund_sqm

    # add these columns to the date_level_features dataframe
    cols_wo_nulls = cols_wo_nulls[cols_wo_nulls != "timestamp"]
    date_level_features = date_level_features.merge(
        macro_df_2013_2015[cols_wo_nulls], on="date", how="left"
    )

    # Date of a PS4 Game Release and Number of Games Released on Date

    # create column for date of a PS4 game release and column for number of games released on date
    ps4games_before_Nov2015 = ps4games[
        ps4games["Release date EU"] <= datetime.datetime(*LAST_DAY_OF_TRAIN_PRD)
    ][["Title", "Genre", "Release date EU", "Addons"]]

    ps4games_before_Nov2015.rename(
        columns={"Release date EU": "release_dt"}, inplace=True
    )

    ps4_game_release_dts = (
        ps4games_before_Nov2015.groupby("release_dt")
        .size()
        .reset_index()
        .rename(columns={"release_dt": "date", 0: "ps4_games_released_cnt"})
    )

    date_level_features["ps4_game_release_dt"] = (
        date_level_features["date"].isin(ps4_game_release_dts["date"]).astype(np.int8)
    )

    date_level_features = date_level_features.merge(
        ps4_game_release_dts, on="date", how="left"
    )
    date_level_features["ps4_games_released_cnt"].fillna(0, inplace=True)

    # Flag for First 3 Days from Game Release Date

    # also create a column flagging first 3 days from game release date (inclusive):
    release_dates_plus2 = (
        date_level_features.query("ps4_game_release_dt == 1")["date"]
        .apply(pd.date_range, periods=3, freq="D")
        .explode()
    )
    date_level_features["ps4_game_release_dt_plus_2"] = (
        date_level_features["date"].isin(release_dates_plus2).astype(np.int8)
    )

    # Time Series Autocorrelations and Cross Correlations

    # add daily quantity column
    date_level_features["day_total_qty_sold"] = (
        sales.groupby("date").item_cnt_day.sum().values
    )

    # create columns for 1-day, 6-day and 7-day lagged total quantity sold
    for shift_val in [1, 6, 7]:
        date_level_features[
            f"day_total_qty_sold_{shift_val}day_lag"
        ] = date_level_features.day_total_qty_sold.shift(shift_val).fillna(0)

    # create column for 1-day lagged brent price (based on the results of cross-correlation analysis)
    date_level_features["brent_1day_lag"] = date_level_features.brent.shift(1).fillna(0)

    date_level_features = _downcast(date_level_features)
    date_level_features = _add_col_prefix(date_level_features, "d_")

    logging.info(
        f"Date-level dataframe has {date_level_features.shape[0]} rows and "
        f"{date_level_features.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Date-level dataframe has the following columns: "
        f"{nl}{nl.join(date_level_features.columns.to_list())}"
    )
    logging.info(
        f"Date-level dataframe has the following data types: "
        f"{nl}{date_level_features.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = date_level_features.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Date-level dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    if to_sql:
        start_instance()
        date_level_features.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(date_level_features)
        write_df_to_sql(date_level_features, "dates", dtypes_dict)
        stop_instance()

    if return_df:
        return date_level_features


# ITEM-DATE-LEVEL FEATURES
def _drange(date_ser):
    """Create complete pandas Series of dates between first and last date in Series passed to the function.

    Parameters:
    -----------
    date_ser : pandas Series
        Series of datetime values with or without any gaps in dates

    Returns:
    --------
    pandas Series of all dates between first and last dates in Series passed to the function
    """
    s = date_ser.min()
    e = date_ser.max()
    return pd.Series(pd.date_range(s, e))


def _lag_merge_asof(df, col_to_count, lag):
    """Compute number of unique values in specified column over all prior dates
    up to the date 'lag' days before current date (inclusive).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe at col_to_count - date level, sorted by date
    col_to_count : str
        Name of column with values that need to be counted
    lag : int
        Number of days before current date at which to end lookback window.
        Example: If lag=1, number of unique values is calculated across all dates
        before (i.e., not including) current date.

    Returns:
    --------
    Original dataframe with a column added for number of unique shop_id values
    and with that column containing null values for item-dates for which
    no lookback period could be constructed given the data.
    """
    d = df.set_index("date")[col_to_count].expanding().apply(lambda x: len(set(x)))
    d.index = d.index + pd.offsets.Day(lag)
    d = d.reset_index(name="num_unique_values_prior_to_day")
    return pd.merge_asof(df, d)


def _spike_check(arr):
    """Determine if array, with last element any non-positive elements removed,
    has any outlier values (and how many).

    Parameters:
    -----------
    arr : array (1d)
        Array to check for outliers

    Returns:
    --------
    int
        Number of elements in array that can be classified as outliers.
        If all data elements in array being evaluated are masked, the function
        returns 0 (which means that none of the elements in that array are
        outliers).
    """
    masked_arr = np.ma.MaskedArray(arr[:-1], mask=(np.array(arr[:-1] <= 0)))
    if isinstance(
        (masked_arr - np.ma.median(masked_arr) > 2 * np.ma.std(masked_arr)).sum(),
        np.ma.core.MaskedConstant,
    ):
        return 0
    return (masked_arr - np.ma.median(masked_arr) > 2 * np.ma.std(masked_arr)).sum()


def _mode(ser):
    """Compute the mode of a pandas Series, with the largest value chosen if
    multiple modes are found.

    Parameters:
    -----------
    ser : pandas Series
        Series for which mode is to be calculated

    Returns:
    --------
    int (scalar)
        Mode of the series
    """
    vc = ser.value_counts()
    return vc[vc == vc.max()].sort_index(ascending=False).index[0]


@Timer(logger=logging.info)
def add_zero_qty_rows(df, levels):
    """Add new (missing) rows between first and last dates of observed sales for
    each value of levels, with 0 values for quantity sold.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new rows
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    all_level_dates = df.groupby([level + "_id" for level in levels]).date.apply(
        _drange
    )
    all_level_dates = all_level_dates.reset_index(
        level=[x[0] for x in enumerate(levels)]
    ).reset_index(drop=True)
    df = all_level_dates.merge(
        df, on=[level + "_id" for level in levels] + ["date"], how="left"
    )
    del all_level_dates
    df[f"{'_'.join(levels)}_qty_sold_day"].fillna(0, inplace=True)
    return _downcast(df)


def _addl_dts(df):
    """Create array of all dates between first and last dates in date column
    of passed datafame.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe with 'date' column that needs to be resampled at daily level

    Returns:
    --------
    numpy 1-d array
    """
    # return df.set_index("date").resample("D").asfreq().reset_index().values.flatten()
    return df.set_index("date").resample("D").asfreq().index.to_numpy()


def _drop_first_row(df):
    """Remove first row from dataframe.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe from which to remove first row

    Returns:
    --------
    pandas DataFrame
    """
    return df.iloc[
        1:,
    ]


@Timer(logger=logging.info)
def add_zero_qty_after_last_date_rows(df, levels):
    """Add new (missing) rows between last observed sale date and last day of
    the training period for each value of levels and only for levels that exist
    in the test data, with 0 values for quantity sold.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new rows
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    test_levels = (
        test_df[[level + "_id" for level in levels]]
        .drop_duplicates()
        .sort_values(by=[level + "_id" for level in levels])
        .reset_index(drop=True)
    )
    last_levels_dts_in_train_data = (
        df.groupby([level + "_id" for level in levels])
        .date.max()
        .reset_index(name=f"last_{'_'.join(levels)}_date")
        # .rename(columns={"date": f"last_{'_'.join(levels)}_date"})
    )
    last_levels_dts_in_train_data["last_train_dt"] = datetime.datetime(
        *FIRST_DAY_OF_TEST_PRD
    )
    addl_dates = (
        last_levels_dts_in_train_data.set_index([level + "_id" for level in levels])
        .stack()
        .reset_index(level=len(levels), drop=True)
        .to_frame()
        .rename(columns={0: "date"})
    )
    del last_levels_dts_in_train_data

    results = []
    for i, (g, grp) in enumerate(addl_dates.groupby(addl_dates.index)):
        if i % 100 == 0:
            gc.collect()
        if len(levels) == 2:
            results.append((*g, _addl_dts(grp)))
        else:
            results.append((g, _addl_dts(grp)))

    addl_dates = _downcast(
        pd.DataFrame(results, columns=[level + "_id" for level in levels] + ["date"])
    )
    logging.debug("Starting to run explode() in add_zero_qty_after_last_date_rows()")
    addl_dates = addl_dates.explode("date").reset_index(drop=True)
    del results

    first_day = datetime.datetime(*FIRST_DAY_OF_TEST_PRD)
    logging.debug("Starting to run query() in add_zero_qty_after_last_date_rows()")
    addl_dates.query("date != @first_day", inplace=True)

    logging.debug("Starting to run _drop_first_row in add_zero_qty_after_last_date_rows()")
    results = []
    for i, (g, grp) in enumerate(
        addl_dates.groupby([level + "_id" for level in levels])
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_drop_first_row(grp))
    addl_dates = pd.concat(results, ignore_index=True)
    del results

    logging.debug("Starting to run merge() in add_zero_qty_after_last_date_rows()")
    addl_dates = addl_dates.merge(
        test_levels, on=[level + "_id" for level in levels], how="inner"
    )

    logging.debug("Starting to run concat() in add_zero_qty_after_last_date_rows()")
    df = pd.concat([df, addl_dates], axis=0, ignore_index=True)
    del addl_dates
    logging.debug("Starting to run sort_values() in add_zero_qty_after_last_date_rows()")
    df.sort_values(
        by=[level + "_id" for level in levels] + ["date"],
        inplace=True,
        ignore_index=True,
    )
    logging.debug("Starting to run fillna() in add_zero_qty_after_last_date_rows()")
    df[f"{'_'.join(levels)}_qty_sold_day"].fillna(0, inplace=True)
    return _downcast(df)


@Timer(logger=logging.info)
def prev_nonzero_qty_sold(df, levels):
    """Create previous non-zero quantity sold column in dataframe, grouped by values in
    specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    df[f"{'_'.join(levels)}_last_qty_sold"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .replace(to_replace=0, method="ffill")
        .shift()
    )
    df.loc[
        df.groupby([level + "_id" for level in levels])[
            f"{'_'.join(levels)}_last_qty_sold"
        ]
        .head(1)
        .index,
        f"{'_'.join(levels)}_last_qty_sold",
    ] = np.NaN
    # fill null values (first item-date) with 0's
    df[f"{'_'.join(levels)}_last_qty_sold"].fillna(0, inplace=True)
    return _downcast(df)


@Timer(logger=logging.info)
def days_elapsed_since_prev_sale(df, levels):
    """Create days elapsed since previous sale column in dataframe, grouped by values in
    specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    last_date = f"{'_'.join(levels)}_last_date"

    df[last_date] = np.where(df[f"{'_'.join(levels)}_qty_sold_day"] > 0, df.date, None,)
    df[last_date] = pd.to_datetime(df[last_date])
    df[last_date] = df[last_date].fillna(method="ffill").shift()
    df.loc[
        df.groupby([level + "_id" for level in levels])[last_date].head(1).index,
        last_date,
    ] = pd.NaT
    df[f"{'_'.join(levels)}_days_since_prev_sale"] = df.date.sub(df[last_date]).dt.days
    df[f"{'_'.join(levels)}_days_since_prev_sale"].fillna(0, inplace=True)
    df.drop(last_date, axis=1, inplace=True)
    return _downcast(df)


@Timer(logger=logging.info)
def days_elapsed_since_first_sale(df, levels):
    """Create days elapsed since first sale column in dataframe, grouped by values in
    specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    df[f"{'_'.join(levels)}_days_since_first_sale"] = (
        df["date"]
        - df.groupby([level + "_id" for level in levels])["date"].transform("first")
    ).dt.days
    return _downcast(df)


@Timer(logger=logging.info)
def first_week_month_of_sale(df, levels):
    """Create indicator columns for first week and first month of sale.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    # create indicator column for first week of sale
    df[f"{'_'.join(levels)}_first_week"] = (
        df[f"{'_'.join(levels)}_days_since_first_sale"] <= 6
    ).astype(np.int8)
    # create indicator column for first month of sale
    df[f"{'_'.join(levels)}_first_month"] = (
        df[f"{'_'.join(levels)}_days_since_first_sale"] <= 30
    ).astype(np.int8)
    return _downcast(df)


def _n_sale_dts(df, levels):
    """Helper function used in creating new columns in dataframe for count of
    sales dates in specified periods before current date.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    pandas DataFrame
    """
    df[f"{'_'.join(levels)}_cnt_sale_dts_last_7d"] = (
        df["day_w_sale"].rolling(7, min_periods=1).sum().shift().fillna(0)
    )
    df[f"{'_'.join(levels)}_cnt_sale_dts_last_30d"] = (
        df["day_w_sale"].rolling(30, min_periods=1).sum().shift().fillna(0)
    )
    df[f"{'_'.join(levels)}_cnt_sale_dts_before_day"] = (
        df["day_w_sale"].expanding().sum().shift().fillna(0)
    )

    return df.drop("day_w_sale", axis=1)


@Timer(logger=logging.info)
def num_of_sale_dts_in_prev_x_days(df, levels, to_sql=False):
    """Create columns for number of days in previous 7-day and 30-day periods
    with a sale, as well as number of days with a sale since start of train period.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    df["day_w_sale"] = np.where(df[f"{'_'.join(levels)}_qty_sold_day"] > 0, 1, 0)

    results = []
    for i, (g, grp) in enumerate(
        df[[level + "_id" for level in levels] + ["date"] + ["day_w_sale"]].groupby(
            [level + "_id" for level in levels]
        )
    ):
        if i % 25 == 0:
            gc.collect()
        results.append(_n_sale_dts(grp, levels))
    n_sale_dts_df = pd.concat(results, ignore_index=True)
    del results

    # For larger data (shop-item-date level), upload results directly to SQL table,
    # instead of merging with other columns
    if len(levels) == 2:
        n_sale_dts_df = _downcast(n_sale_dts_df)
        n_sale_dts_df = _add_col_prefix(n_sale_dts_df, "sid_")

        logging.info(
            f"Shop-item-date-level dataframe with columns for number of sale "
            f"dates in previous x days has {n_sale_dts_df.shape[0]} rows and "
            f"{n_sale_dts_df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-item-date-level dataframe with columns for number of sale "
            f"dates in previous x days has the following columns: "
            f"{nl}{nl.join(n_sale_dts_df.columns.to_list())}"
        )
        logging.info(
            f"Shop-item-date-level dataframe with columns for number of sale "
            f"dates in previous x days has the following data types: "
            f"{nl}{n_sale_dts_df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = n_sale_dts_df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-item-date-level dataframe with columns for number of sale "
                f"dates in previous x days has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        if to_sql:
            start_instance()
            n_sale_dts_df.rename(columns={"date": "sale_date"}, inplace=True)
            dtypes_dict = _map_to_sql_dtypes(n_sale_dts_df)
            # add dictionary of data types in n_sale_dts_df to master dictionary
            # of data types in dataframes to be written to SQL and to be queried
            # later
            # first, check if master dictionary file already exists
            # if not, run initialization code
            input_json = "master_pd_types.json"
            types_json = Path(input_json)
            if types_json.is_file():
                with open(input_json, "r") as fp:
                    master_pd_types = json.load(fp)
            else:
                build_item_lvl_features()
                with open(input_json, "r") as fp:
                    master_pd_types = json.load(fp)
            master_pd_types.update(
                {"n_sale_dts_df": n_sale_dts_df.dtypes.map(str).to_dict()}
            )
            with open("master_pd_types.json", "w") as fp:
                json.dump(master_pd_types, fp)
            write_df_to_sql(n_sale_dts_df, "sid_n_sale_dts", dtypes_dict)
            stop_instance()
            del n_sale_dts_df

    else:
        df = df.merge(
            n_sale_dts_df, on=[level + "_id" for level in levels] + ["date"], how="left"
        )
        del n_sale_dts_df

        df.drop("day_w_sale", axis=1, inplace=True)

    return _downcast(df)


def _rolling_stats(df, levels):
    """Helper function used in creating columns for rolling max, min, mean, mode
    and median quantity sold values.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    pandas DataFrame
    """
    df[f"{'_'.join(levels)}_rolling_7d_max_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .rolling(7, 1)
        .max()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_rolling_7d_min_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .rolling(7, 1)
        .min()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_rolling_7d_avg_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .rolling(7, 1)
        .mean()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_rolling_7d_mode_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .rolling(7, 1)
        .apply(lambda x: _mode(x))
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_rolling_7d_median_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .rolling(7, 1)
        .median()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )

    return df.drop(f"{'_'.join(levels)}_qty_sold_day", axis=1)


@Timer(logger=logging.info)
def rolling_7d_qty_stats(df, levels, to_sql=False):
    """Create rolling max, min, mean, mode and median quantity sold values,
    grouped by values of specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    results = []
    for i, (g, grp) in enumerate(
        df[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_qty_sold_day"]
        ].groupby([level + "_id" for level in levels])
    ):
        if i % 25 == 0:
            gc.collect()
        results.append(_rolling_stats(grp, levels))
    roll_7d_qty_df = pd.concat(results, ignore_index=True)
    del results

    # For larger data (shop-item-date level), upload results directly to SQL table,
    # instead of merging with other columns
    if len(levels) == 2:
        roll_7d_qty_df = _downcast(roll_7d_qty_df)
        roll_7d_qty_df = _add_col_prefix(roll_7d_qty_df, "sid_")

        logging.info(
            f"Shop-item-date-level dataframe with columns for rolling 7-day "
            f"quantity stats has {roll_7d_qty_df.shape[0]} rows and "
            f"{roll_7d_qty_df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-item-date-level dataframe with columns for rolling 7-day "
            f"quantity stats has the following columns: "
            f"{nl}{nl.join(roll_7d_qty_df.columns.to_list())}"
        )
        logging.info(
            f"Shop-item-date-level dataframe with columns for rolling 7-day "
            f"quantity stats has the following data types: "
            f"{nl}{roll_7d_qty_df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = roll_7d_qty_df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-item-date-level dataframe with columns for rolling 7-day "
                f"quantity stats has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        if to_sql:
            start_instance()
            roll_7d_qty_df.rename(columns={"date": "sale_date"}, inplace=True)
            dtypes_dict = _map_to_sql_dtypes(roll_7d_qty_df)
            write_df_to_sql(roll_7d_qty_df, "sid_roll_qty_stats", dtypes_dict)
            stop_instance()
            del roll_7d_qty_df

    else:
        df = df.merge(
            roll_7d_qty_df,
            on=[level + "_id" for level in levels] + ["date"],
            how="left",
        )
        del roll_7d_qty_df

    return _downcast(df)


def _expand_cv2(df, levels):
    """Helper function used in creating column for expanding coefficient of
    variation squared of quantity bought before current day, with only positive
    quantity values considered.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    pandas DataFrame
    """
    df["expand_cv2_of_qty"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .apply(
            lambda x: np.square(
                np.ma.std(np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) <= 0)))
                / np.ma.mean(np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) <= 0)))
            ),
            raw=True
            # engine="numba",
            # engine_kwargs={"nopython": False},
        )
        .fillna(0)
        .values
    )
    return df.drop(f"{'_'.join(levels)}_qty_sold_day", axis=1)


@Timer(logger=logging.info)
def expanding_cv2_of_qty(df, levels, to_sql=False):
    """Create column for expanding coefficient of variation squared of quantity
    bought before current day, with only positive quantity values considered.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    results = []
    for i, (g, grp) in enumerate(
        df[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_qty_sold_day"]
        ].groupby([level + "_id" for level in levels])
    ):
        if i % 25 == 0:
            gc.collect()
        results.append(_expand_cv2(grp, levels))
    expand_qty_cv2_df = pd.concat(results, ignore_index=True)
    del results

    # For larger data (shop-item-date level), upload results directly to SQL table,
    # instead of merging with other columns
    if len(levels) == 2:
        expand_qty_cv2_df = _downcast(expand_qty_cv2_df)
        expand_qty_cv2_df = _add_col_prefix(expand_qty_cv2_df, "sid_")

        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding cv2 "
            f"of quantity has {expand_qty_cv2_df.shape[0]} rows and "
            f"{expand_qty_cv2_df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding cv2 "
            f"of quantity has the following columns: "
            f"{nl}{nl.join(expand_qty_cv2_df.columns.to_list())}"
        )
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding cv2 "
            f"of quantity has the following data types: "
            f"{nl}{expand_qty_cv2_df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = expand_qty_cv2_df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-item-date-level dataframe with columns for expanding cv2 "
                f"of quantity has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        if to_sql:
            start_instance()
            expand_qty_cv2_df.rename(columns={"date": "sale_date"}, inplace=True)
            dtypes_dict = _map_to_sql_dtypes(expand_qty_cv2_df)
            write_df_to_sql(expand_qty_cv2_df, "sid_expand_qty_cv_sqrd", dtypes_dict)
            stop_instance()
            del expand_qty_cv2_df

    else:
        df = df.merge(
            expand_qty_cv2_df,
            on=[level + "_id" for level in levels] + ["date"],
            how="left",
        )
        del expand_qty_cv2_df

    return _downcast(df)


@Timer(logger=logging.info)
def expanding_avg_demand_int(df, levels):
    """Create expanding average demand interval before current day column.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    if len(levels) == 2:
        start_instance()
        sql_str = "SELECT " "{0} " "FROM sid_n_sale_dts " "ORDER BY {1};"
        sql = SQL(sql_str).format(
            SQL(", ").join(
                [
                    Identifier(col)
                    for col in [level + "_id" for level in levels]
                    + ["sale_date"]
                    + [f"sid_{'_'.join(levels)}_cnt_sale_dts_before_day"]
                ]
            ),
            SQL(", ").join(
                [
                    Identifier(col)
                    for col in [level + "_id" for level in levels] + ["sale_date"]
                ]
            ),
        )
        # pass master dictionary of data types to df_from_sql_query function
        input_json = "master_pd_types.json"
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)
        sale_dts_df = df_from_sql_query(
            sql, master_pd_types["n_sale_dts_df"], date_list=["sale_date"]
        )
        df[f"{'_'.join(levels)}_expanding_adi"] = (
            df[f"{'_'.join(levels)}_days_since_first_sale"]
            .div(sale_dts_df[f"sid_{'_'.join(levels)}_cnt_sale_dts_before_day"])
            .fillna(0)
        )
        stop_instance()

    else:
        df[f"{'_'.join(levels)}_expanding_adi"] = (
            df[f"{'_'.join(levels)}_days_since_first_sale"]
            .div(df[f"{'_'.join(levels)}_cnt_sale_dts_before_day"])
            .fillna(0)
        )

    return _downcast(df)


def _expanding_stats(df, levels):
    """Helper function used in creating expanding max, min, mean, mode and median
    quantity sold values, grouped by values of specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    pandas DataFrame
    """
    df[f"{'_'.join(levels)}_expand_qty_max"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .max()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_expand_qty_min"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .min()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_expand_qty_mean"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .mean()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_expand_qty_mode"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .apply(lambda x: _mode(x))
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )
    df[f"{'_'.join(levels)}_expand_qty_median"] = (
        df[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .median()
        .shift()
        .fillna(df[f"{'_'.join(levels)}_qty_sold_day"])
    )

    return df.drop(f"{'_'.join(levels)}_qty_sold_day", axis=1)


@Timer(logger=logging.info)
def expanding_qty_sold_stats(df, levels, to_sql=False):
    """Create expanding max, min, mean, mode and median quantity sold values,
    grouped by values of specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    results = []
    for i, (g, grp) in enumerate(
        df[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_qty_sold_day"]
        ].groupby([level + "_id" for level in levels])
    ):
        if i % 25 == 0:
            gc.collect()
        results.append(_expanding_stats(grp, levels))
    expand_qty_stats_df = pd.concat(results, ignore_index=True)
    del results

    # For larger data (shop-item-date level), upload results directly to SQL table,
    # instead of merging with other columns
    if len(levels) == 2:
        expand_qty_stats_df = _downcast(expand_qty_stats_df)
        expand_qty_stats_df = _add_col_prefix(expand_qty_stats_df, "sid_")

        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding "
            f"quantity stats has {expand_qty_stats_df.shape[0]} rows and "
            f"{expand_qty_stats_df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding "
            f"quantity stats has the following columns: "
            f"{nl}{nl.join(expand_qty_stats_df.columns.to_list())}"
        )
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding "
            f"quantity stats has the following data types: "
            f"{nl}{expand_qty_stats_df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = expand_qty_stats_df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-item-date-level dataframe with columns for expanding "
                f"quantity stats has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        if to_sql:
            start_instance()
            expand_qty_stats_df.rename(columns={"date": "sale_date"}, inplace=True)
            dtypes_dict = _map_to_sql_dtypes(expand_qty_stats_df)
            write_df_to_sql(expand_qty_stats_df, "sid_expand_qty_stats", dtypes_dict)
            stop_instance()
            del expand_qty_stats_df

    else:
        df = df.merge(
            expand_qty_stats_df,
            on=[level + "_id" for level in levels] + ["date"],
            how="left",
        )
        del expand_qty_stats_df

    return _downcast(df)


@Timer(logger=logging.info)
def qty_sold_x_days_before(df, levels):
    """Create columns for quantity 1, 2, 3 days ago.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    for shift_val in [1, 2, 3, 7]:
        df[f"{'_'.join(levels)}_qty_sold_{shift_val}d_ago"] = df.groupby(
            [level + "_id" for level in levels]
        )[f"{'_'.join(levels)}_qty_sold_day"].shift(shift_val)
        df[f"{'_'.join(levels)}_qty_sold_{shift_val}d_ago"].fillna(0, inplace=True)
    return _downcast(df)


def _expanding_bw_sales_stats(df, levels):
    """Helper function used in creating expanding max, min, mean, mode, median
    and standard deviation of days between sales columns, grouped by values of
    specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    pandas DataFrame
    """
    df[f"{'_'.join(levels)}_date_max_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .max()
        .shift()
        .fillna(0)
        .astype('uint16')
    )
    df[f"{'_'.join(levels)}_date_min_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .min()
        .shift()
        .fillna(0)
        .astype('uint16')
    )
    df[f"{'_'.join(levels)}_date_avg_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .mean()
        .shift()
        .fillna(0)
        .astype('float32')
    )
    df[f"{'_'.join(levels)}_date_median_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .median()
        .shift()
        .fillna(0)
        .astype('float32')
    )
    df[f"{'_'.join(levels)}_date_mode_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .apply(lambda x: _mode(x))
        .shift()
        .fillna(0)
        .astype('uint16')
    )
    df[f"{'_'.join(levels)}_date_std_gap_bw_sales"] = (
        df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        .expanding()
        .std(ddof=0)
        .shift()
        .fillna(0)
        .astype('float32')
    )

    return df.drop(f"{'_'.join(levels)}_days_since_prev_sale_lmtd", axis=1)


@Timer(logger=logging.info)
def expanding_time_bw_sales_stats(df, levels, to_sql=False):
    """Create expanding max, min, mean, mode, median and standard deviation of
    days between sales columns, grouped by values of specified column(s).

    Note:
    -----
    Requires days_elapsed_since_prev_sale() to be run prior in the same call
    to build_... function

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"] = np.where(
        df[f"{'_'.join(levels)}_qty_sold_day"] > 0,
        df[f"{'_'.join(levels)}_days_since_prev_sale"],
        np.nan,
    )

    results = []
    for i, (g, grp) in enumerate(
        df[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_days_since_prev_sale_lmtd"]
        ].groupby([level + "_id" for level in levels])
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_expanding_bw_sales_stats(grp, levels))
    expand_bw_sales_stats_df = pd.concat(results, ignore_index=True)
    del results

    # For larger data (shop-item-date level), upload results directly to SQL table,
    # instead of merging with other columns
    if len(levels) == 2:
        expand_bw_sales_stats_df = _downcast(expand_bw_sales_stats_df)
        expand_bw_sales_stats_df = _add_col_prefix(expand_bw_sales_stats_df, "sid_")

        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding days "
            f"between sales stats has {expand_bw_sales_stats_df.shape[0]} rows and "
            f"{expand_bw_sales_stats_df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding days "
            f"between sales stats has the following columns: "
            f"{nl}{nl.join(expand_bw_sales_stats_df.columns.to_list())}"
        )
        logging.info(
            f"Shop-item-date-level dataframe with columns for expanding days "
            f"between sales stats has the following data types: "
            f"{nl}{expand_bw_sales_stats_df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = expand_bw_sales_stats_df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-item-date-level dataframe with columns for expanding days "
                f"between sales stats has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        if to_sql:
            start_instance()
            expand_bw_sales_stats_df.rename(columns={"date": "sale_date"}, inplace=True)
            dtypes_dict = _map_to_sql_dtypes(expand_bw_sales_stats_df)
            write_df_to_sql(
                expand_bw_sales_stats_df, "sid_expand_bw_sales_stats", dtypes_dict
            )
            stop_instance()
            del expand_bw_sales_stats_df

    else:
        df = df.merge(
            expand_bw_sales_stats_df,
            on=[level + "_id" for level in levels] + ["date"],
            how="left",
        )
        del expand_bw_sales_stats_df
        df.drop(f"{'_'.join(levels)}_days_since_prev_sale_lmtd", axis=1, inplace=True)

    return _downcast(df)


@Timer(logger=logging.info)
def diff_bw_last_and_sec_to_last_qty(df, levels):
    """Create column containing difference between last quantity sold and
    second-to-last quantity sold, grouped by values of specified column(s).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    non_zero_qty_level_dates = _downcast(
        df.query(f"{'_'.join(levels)}_qty_sold_day != 0")[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_qty_sold_day"]
        ]
    )
    last_date_per_level = _downcast(
        df[
            [level + "_id" for level in levels]
            + ["date"]
            + [f"{'_'.join(levels)}_qty_sold_day"]
        ]
        .groupby([level + "_id" for level in levels])
        .tail(1)
        .reset_index(drop=True)
    )
    last_date_per_level.query(f"{'_'.join(levels)}_qty_sold_day == 0", inplace=True)
    last_date_per_level["date"] = datetime.datetime(*FIRST_DAY_OF_TEST_PRD)
    last_date_per_level[f"{'_'.join(levels)}_qty_sold_day"] = 10
    non_zero_qty_level_dates = pd.concat(
        [non_zero_qty_level_dates, last_date_per_level], axis=0, ignore_index=True
    )
    del last_date_per_level
    non_zero_qty_level_dates.sort_values(
        by=[level + "_id" for level in levels] + ["date"],
        inplace=True,
        ignore_index=True,
    )
    non_zero_qty_level_dates[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"] = (
        non_zero_qty_level_dates.groupby([level + "_id" for level in levels])[
            f"{'_'.join(levels)}_qty_sold_day"
        ]
        .diff(periods=2)
        .values
        - non_zero_qty_level_dates.groupby([level + "_id" for level in levels])[
            f"{'_'.join(levels)}_qty_sold_day"
        ]
        .diff()
        .values
    )
    non_zero_qty_level_dates.drop(
        f"{'_'.join(levels)}_qty_sold_day", axis=1, inplace=True
    )
    non_zero_qty_level_dates[
        f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"
    ].fillna(0, inplace=True)

    first_day = datetime.datetime(*FIRST_DAY_OF_TEST_PRD)

    # For larger data (shop-item-date and item-date level), upload intermediate
    # results to SQL tables and concatenate and merge in-database
    if (len(levels) == 2) | (levels == ["item"]):
        # export dataframes to SQL tables before concatenating to save memory
        start_instance()
        df = _downcast(df)
        df.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(df)

        input_json = "master_pd_types.json"
        types_json = Path(input_json)
        if types_json.is_file():
            with open(input_json, "r") as fp:
                master_pd_types = json.load(fp)
        else:
            build_item_lvl_features()
            with open(input_json, "r") as fp:
                master_pd_types = json.load(fp)
        master_pd_types.update({"df": df.dtypes.map(str).to_dict()})

        write_df_to_sql(df, "df_temp", dtypes_dict)

        non_zero_qty_level_dates = _downcast(non_zero_qty_level_dates)
        non_zero_qty_level_dates.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(non_zero_qty_level_dates)
        master_pd_types.update(
            {
                "non_zero_qty_level_dates": non_zero_qty_level_dates.dtypes.map(
                    str
                ).to_dict()
            }
        )
        write_df_to_sql(non_zero_qty_level_dates, "non_zero_temp", dtypes_dict)

        # per https://www.psycopg.org/docs/sql.html#psycopg2.sql.SQL
        # UNION: number and order of columns must be the same and data types
        # must be compatible
        df_cols = df.columns.sort_values().to_list()
        non_zero_cols = non_zero_qty_level_dates.columns.to_list()
        logging.debug(
            f"DF dataframe has {df.shape[0]} rows and "
            f"{df.shape[1]} columns."
        )
        logging.debug(
            f"non_zero_qty_level_dates dataframe has {non_zero_qty_level_dates.shape[0]} rows and "
            f"{non_zero_qty_level_dates.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        miss_vls = df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"DF dataframe before multi-part JOIN has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )
        miss_vls = non_zero_qty_level_dates.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"non_zero_qty_level_dates dataframe before multi-part JOIN has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

        del df
        del non_zero_qty_level_dates

        if levels == ["item"]:
            sql_str = (
                "SELECT * FROM ( "
                # f
                "SELECT *, "
                    "CASE WHEN {9} IS NOT NULL THEN {9} ELSE MAX({9}) "
                    "OVER (PARTITION BY item_id, ctr ORDER BY sale_date DESC) END diff_filled_in "
                    # e
                    "FROM (SELECT *, "
                        "SUM(CASE WHEN {9} IS NOT NULL THEN 1 ELSE 0 END) OVER "
                        "(PARTITION BY item_id ORDER BY sale_date DESC) ctr "
                            # d
                            "FROM (SELECT b.*, {8} "
                                "FROM (SELECT * "
                                    "FROM (SELECT {0}, {1} FROM df_temp) AS a "
                                        "UNION ALL (SELECT {0}, {2} "
                                        "FROM non_zero_temp "
                                        "WHERE sale_date = %(dt)s)) AS b "
                                        "LEFT JOIN (SELECT * "
                                        "FROM non_zero_temp) AS c "
                                        "ON {3} = {4} AND {5} = {6} "
                            "ORDER BY {7}) AS d "
                    ") AS e "
                ") AS f "
                "WHERE sale_date <> %(dt)s "
                "ORDER BY {10};"
            )

            # sql_str = (
            #     "SELECT b.*, {8} "
            #     "FROM (SELECT * "
            #     "FROM (SELECT {0}, {1} from df_temp) AS a "
            #     "UNION ALL (SELECT {0}, {2} "
            #     "FROM non_zero_temp "
            #     "WHERE sale_date = %(dt)s)) AS b "
            #     "LEFT JOIN (SELECT * "
            #     "FROM non_zero_temp) AS c "
            #     "ON {3} = {4} AND {5} = {6} "
            #     "ORDER BY {7};"
            # )
            sql = SQL(sql_str).format(
                # 0: columns that exist in both tables
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname in non_zero_cols
                        ]
                    ]
                ),
                # 1: columns that only exist in main df
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname not in non_zero_cols
                        ]
                    ]
                ),
                # 2: columns that only exist in main df, with "null as" added
                SQL("%(none)s AS ")
                + SQL(", %(none)s AS ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname not in non_zero_cols
                        ]
                    ]
                ),
                # 3-6: columns to merge on, with table identifiers
                Identifier("b", "item_id"),
                Identifier("c", "item_id"),
                Identifier("b", "sale_date"),
                Identifier("c", "sale_date"),
                # 7: shop_id, item_id, sale_date - columns to sort final table by
                # [Identifier('b', 'shop_id'), Identifier('b', 'item_id'), Identifier('b', 'sale_date')]
                SQL(", ").join(
                    [
                        Identifier("b", col)
                        for col in [level + "_id" for level in levels] + ["sale_date"]
                    ]
                ),
                Identifier("c", f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"),
                Identifier(f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"),
                # 10:
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [level + "_id" for level in levels] + ["sale_date"]
                    ]
                ),
            )

        else:
            # shift 7, 8, 9 by 2 to get 9, 10, 11
            # add 12 (ORDER BY {10} to ORDER BY {12})
            db_table_name = "sid_big_query_result"
            sql_str = (
                "CREATE TABLE {13} AS SELECT * FROM ( "
                # f
                "SELECT *, "
                    "CASE WHEN {11} IS NOT NULL THEN {11} ELSE MAX({11}) "
                    "OVER (PARTITION BY shop_id, item_id, ctr ORDER BY sale_date DESC) END diff_filled_in "
                    # e
                    "FROM (SELECT *, "
                        "SUM(CASE WHEN {11} IS NOT NULL THEN 1 ELSE 0 END) OVER "
                        "(PARTITION BY shop_id, item_id ORDER BY sale_date DESC) ctr "
                            # d
                            "FROM (SELECT b.*, {10} "
                                "FROM (SELECT * "
                                    "FROM (SELECT {0}, {1} FROM df_temp) AS a "
                                        "UNION ALL (SELECT {0}, {2} "
                                        "FROM non_zero_temp "
                                        "WHERE sale_date = %(dt)s)) AS b "
                                        "LEFT JOIN (SELECT * "
                                        "FROM non_zero_temp) AS c "
                                        "ON {3} = {4} AND {5} = {6} AND {7} = {8} "
                            "ORDER BY {9}) AS d "
                    ") AS e "
                ") AS f "
                "WHERE sale_date <> %(dt)s "
                "ORDER BY {12};"
            )
            # sql_str = (
            #     "SELECT b.*, {10} "
            #     "FROM (SELECT * "
            #     "FROM (SELECT {0}, {1} from df_temp) AS a "
            #     "UNION ALL (SELECT {0}, {2} "
            #     "FROM non_zero_temp "
            #     "WHERE sale_date = %(dt)s)) AS b "
            #     "LEFT JOIN (SELECT * "
            #     "FROM non_zero_temp) AS c "
            #     "ON {3} = {4} AND {5} = {6} AND {7} = {8} "
            #     "ORDER BY {9};"
            # )
            sql = SQL(sql_str).format(
                # 0: columns that exist in both tables
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname in non_zero_cols
                        ]
                    ]
                ),
                # 1: columns that only exist in main df
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname not in non_zero_cols
                        ]
                    ]
                ),
                # 2: columns that only exist in main df, with "null as" added
                SQL("%(none)s AS ")
                + SQL(", %(none)s AS ").join(
                    [
                        Identifier(col)
                        for col in [
                            colname for colname in df_cols if colname not in non_zero_cols
                        ]
                    ]
                ),
                # 3-8: columns to merge on, with table identifiers
                Identifier("b", "shop_id"),
                Identifier("c", "shop_id"),
                Identifier("b", "item_id"),
                Identifier("c", "item_id"),
                Identifier("b", "sale_date"),
                Identifier("c", "sale_date"),
                # 9: shop_id, item_id, sale_date - columns to sort final table by
                # [Identifier('b', 'shop_id'), Identifier('b', 'item_id'), Identifier('b', 'sale_date')]
                SQL(", ").join(
                    [
                        Identifier("b", col)
                        for col in [level + "_id" for level in levels] + ["sale_date"]
                    ]
                ),
                Identifier("c", f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"),
                Identifier(f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"),
                # 12:
                SQL(", ").join(
                    [
                        Identifier(col)
                        for col in [level + "_id" for level in levels] + ["sale_date"]
                    ]
                ),
                # 13:
                Identifier(db_table_name),
            )

        params = {"db_table": db_table_name, "dt": first_day, "none": None}
        # Update dictionary of pandas data types in DF dataframe for columns that
        # will contain nulls after SQL query

        # master_pd_types["non_zero_qty_level_dates"].update(cols_to_change_to_float)

        # d = {**master_pd_types['non_zero_qty_level_dates'], **master_pd_types['df']}
        # logging.debug(
        #     f"Full dictionary to be passed to df_from_sql_query function: "
        #     f"{d}"
        # )
        # # Full dictionary to be passed to df_from_sql_query function:
        # {
        #     'item_id': 'uint16',
        #     'sale_date': 'datetime64[ns]',
        #     'item_date_diff_bw_last_and_prev_qty': 'float32',
        #     'item_qty_sold_day': 'int16'
        # }

        # Write query results to new SQL table
        create_db_table_from_query(sql, params)

        # input_json = "master_pd_types.json"
        # with open(input_json, "r") as fp:
        #     master_pd_types = json.load(fp)
        cast_dict = {**master_pd_types["non_zero_qty_level_dates"], **master_pd_types["df"]}
        # cols_to_change_to_float = {col: 'float32' for col in [
        #     colname for colname in df_cols if colname not in non_zero_cols
        # ]}
        # cols_to_change_to_float[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"] = 'float32'
        # cast_dict.update(cols_to_change_to_float)
        new_col_to_cast = {"diff_filled_in": "int16"}
        cast_dict.update(new_col_to_cast)
        logging.debug(f"cast_dict is {cast_dict}")

        df = df_from_sql_table(db_table_name, cast_dict, date_list=["sale_date"])
        # If df is successfully created, delete the big db tables from which it was created
        if df is not None:
            drop_tables(["df_temp", "non_zero_temp"] + [db_table_name])

        # df = df_from_sql_query(
        #     sql,
        #     cast_dict,
        #     params=params,
        #     date_list=["sale_date"],
        #     # delete_tables=["df_temp", "non_zero_temp"],
        # )
        df.drop([f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty", "ctr"], axis=1, inplace=True)
        df.rename(columns={"sale_date": "date", "diff_filled_in": f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"}, inplace=True)
        with open("master_pd_types.json", "w") as fp:
            json.dump(master_pd_types, fp)
        logging.debug(
            f"DF dataframe after multi-part JOIN has {df.shape[0]} rows and "
            f"{df.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.debug(
            f"DF dataframe after multi-part JOIN has the following data types: "
            f"{nl}{df.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = df.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"DF dataframe after multi-part JOIN has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )

    else:
        df = pd.concat(
            [
                df,
                non_zero_qty_level_dates.query("date == @first_day")[
                    [level + "_id" for level in levels] + ["date"]
                ],
            ],
            axis=0,
            ignore_index=True,
        )
        df = df.merge(
            non_zero_qty_level_dates,
            on=[level + "_id" for level in levels] + ["date"],
            how="left",
        )
        del non_zero_qty_level_dates

        df[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"] = df.groupby(
            [level + "_id" for level in levels]
        )[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"].fillna(method="bfill")

    # if (len(levels) == 2) | (levels == ["item"]):
    #     df = _downcast(df)
    #     df.rename(columns={"date": "sale_date"}, inplace=True)
    #     dtypes_dict = _map_to_sql_dtypes(df)
    #     master_pd_types.update({"df": df.dtypes.map(str).to_dict()})
    #     with open("master_pd_types.json", "w") as fp:
    #         json.dump(master_pd_types, fp)
    #     write_df_to_sql(df, "df_temp", dtypes_dict)
    #     del df
    #     if levels == ["item"]:
    #         sql_str = "SELECT * FROM df_temp WHERE sale_date <> %(dt)s ORDER BY item_id, sale_date;"
    #     else:
    #         sql_str = "SELECT * FROM df_temp WHERE sale_date <> %(dt)s ORDER BY shop_id, item_id, sale_date;"
    #     sql = SQL(sql_str)
    #     params = {"dt": first_day}
    #     df = df_from_sql_query(
    #         sql,
    #         master_pd_types["df"],
    #         params=params,
    #         date_list=["sale_date"],
    #         delete_tables=["df_temp"],
    #     )
    #     df.rename(columns={"sale_date": "date"}, inplace=True)
    # else:
    if levels == ['shop']:
        df = df.query("date != @first_day").reset_index(drop=True)
    return _downcast(df)


@Timer(logger=logging.info)
def num_of_unique_opp_values(df, sales, level):
    """Create column for number of unique values in column opposite to the
    specified column (i.e., for level='item_id', calculate number of unique shops
    that sold the item prior to current day).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new column (dataframe will be modified inplace)
    sales : sales Dataframe
        Existing shop-item-date level dataframe
    level : str
        Level by which to group computations (e.g., 'item_id')

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe
    """
    col_list = ["shop_id", "item_id", "date"]

    col_name = (
        f"num_unique_"
        f"{[col for col in col_list if col not in (level, 'date')][0].replace('_id','s')}"
        f"_prior_to_day"
    )

    if level == "shop_id":
        group_dates_w_val_cts = (
            sales[col_list]
            .sort_values(by=[level, "date"])
            .set_index(level)
        )
        group_dates_w_val_cts.set_index(
            group_dates_w_val_cts.groupby(level).cumcount().rename("iidx"),
            append=True,
            inplace=True
        )
        group_dates_w_val_cts = (
            group_dates_w_val_cts.groupby(level=level)
            .apply(
                _lag_merge_asof,
                [col for col in col_list if col not in (level, "date")][0],
                lag=1,
            )
            .reset_index(level)
            .reset_index(drop=True)
            .drop_duplicates(subset=[level, "date"])
            .rename(columns={"num_unique_values_prior_to_day": col_name})
            .drop([col for col in col_list if col not in (level, "date")][0], axis=1)
        )

    elif level == "item_id":
        start_instance()
        sql_str = (
            "SELECT {0}, row_number() OVER (PARTITION BY {1} ORDER BY sale_date) AS iidx "
            "FROM sales_cleaned ORDER BY {2};"
        )
        sql = SQL(sql_str).format(
            SQL(", ").join(
                [
                    Identifier("sale_" + col) if col == "date" else Identifier(col)
                    for col in col_list
                ]
            ),
            Identifier(level),
            SQL(", ").join([Identifier(col) for col in [level, "sale_date"]]),
        )
        input_json = "master_pd_types.json"
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)

        group_dates_w_val_cts = df_from_sql_query(
            sql, master_pd_types["sales"], date_list=["sale_date"],
        ).rename(columns={"sale_date": "date"})

        results = []
        for i, (g, grp) in enumerate(
            group_dates_w_val_cts.set_index([level, "iidx"]).groupby(level=level)
        ):
            if i % 25 == 0:
                gc.collect()
            results.append(
                _lag_merge_asof(
                    grp.reset_index(level),
                    [col for col in col_list if col not in (level, "date")][0],
                    lag=1,
                )
                .reset_index(drop=True)
                .drop_duplicates(subset=[level, "date"])
                .rename(columns={"num_unique_values_prior_to_day": col_name})
                .drop(
                    [col for col in col_list if col not in (level, "date")][0], axis=1
                )
            )
        group_dates_w_val_cts = pd.concat(results, ignore_index=True)
        del results

    group_dates_w_val_cts[col_name].fillna(0, inplace=True)

    if level == "item_id":
        df.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(df)

        master_pd_types.update({"df_in_opp_vals_func": df.dtypes.map(str).to_dict()})

        write_df_to_sql(df, "df_temp", dtypes_dict)
        del df

        group_dates_w_val_cts = _downcast(group_dates_w_val_cts)
        group_dates_w_val_cts.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(group_dates_w_val_cts)
        master_pd_types.update(
            {"group_dates_w_val_cts": group_dates_w_val_cts.dtypes.map(str).to_dict()}
        )
        write_df_to_sql(group_dates_w_val_cts, "group_dates_w_val_cts", dtypes_dict)
        del group_dates_w_val_cts

        sql_str = "SELECT {0} " "FROM sales_cleaned ORDER BY {1};"
        sql = SQL(sql_str).format(
            SQL(", ").join(
                [
                    Identifier("sale_" + col) if col == "date" else Identifier(col)
                    for col in col_list
                ]
            ),
            SQL(", ").join(Identifier(col) for col in [level, "sale_date"]),
        )

        daily_cts_wo_lag = df_from_sql_query(
            sql, master_pd_types["sales"], date_list=["sale_date"],
        ).rename(columns={"sale_date": "date"})

        results = []
        for i, (g, grp) in enumerate(daily_cts_wo_lag.groupby(level)):
            if i % 25 == 0:
                gc.collect()
            grp["_daily_cts_wo_lag"] = (
                grp[[col for col in col_list if col not in (level, "date")][0]]
                .expanding()
                .apply(lambda x: len(set(x)))
            )
            results.append(
                grp.drop_duplicates([level, "date"], keep="last")
            )
            # results.append(
            #     grp[[col for col in col_list if col not in (level, "date")][0]]
            #     .expanding()
            #     .apply(lambda x: len(set(x)))
            #     # .reset_index(level=2, drop=True)
            #     .reset_index(name="_daily_cts_wo_lag")
            #     .drop_duplicates([level, "date"], keep="last")
            # )
        daily_cts_wo_lag = pd.concat(results, ignore_index=True)
        del results

        daily_cts_wo_lag = _downcast(daily_cts_wo_lag)
        daily_cts_wo_lag.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(daily_cts_wo_lag)
        master_pd_types.update(
            {"daily_cts_wo_lag": daily_cts_wo_lag.dtypes.map(str).to_dict()}
        )
        write_df_to_sql(daily_cts_wo_lag, "daily_cts_wo_lag", dtypes_dict)
        del daily_cts_wo_lag

        sql_str = (
            "SELECT a.*, {0}, {1} "
            "FROM df_temp AS a "
            "LEFT JOIN group_dates_w_val_cts AS b "
            "ON {2} = {3} AND {5} = {6} "
            "LEFT JOIN daily_cts_wo_lag AS c "
            "ON {2} = {4} AND {5} = {7} "
            "ORDER BY {8}, {9};"
        )
        sql = SQL(sql_str).format(
            # 0: column from group_dates_w_val_cts
            Identifier("b", col_name),
            # 1: column from daily_cts_wo_lag
            Identifier("c", "_daily_cts_wo_lag"),
            # 2-7: columns to merge on
            Identifier("a", level),
            Identifier("b", level),
            Identifier("c", level),
            Identifier("a", "sale_date"),
            Identifier("b", "sale_date"),
            Identifier("c", "sale_date"),
            # 8-9: columns to sort by
            Identifier("a", level),
            Identifier("a", "sale_date"),
        )

        # params = {"lvl": level}
        # update data types of columns that will have null values after SQL join
        master_pd_types["group_dates_w_val_cts"].update({'num_unique_shops_prior_to_day': 'float32'})
        master_pd_types["daily_cts_wo_lag"].update({'_daily_cts_wo_lag': 'float32'})
        df = df_from_sql_query(
            sql,
            {
                **master_pd_types["df_in_opp_vals_func"],
                **master_pd_types["group_dates_w_val_cts"],
                **master_pd_types["daily_cts_wo_lag"],
            },
            # params=params,
            date_list=["sale_date"],
            delete_tables=["df_temp", "group_dates_w_val_cts", "daily_cts_wo_lag"],
        )
        df.rename(columns={"sale_date": "date"}, inplace=True)

        # save the updated JSON file
        with open("master_pd_types.json", "w") as fp:
            json.dump(master_pd_types, fp)

    elif level == "shop_id":
        df = df.merge(group_dates_w_val_cts, on=[level, "date"], how="left")

        # fill null values on days when no sale was made
        daily_cts_wo_lag = (
            sales[col_list]
            .sort_values(by=[level, "date"], ignore_index=True)
            .groupby([level, "date"])[
                [col for col in col_list if col not in (level, "date")][0]
            ]
            .expanding()
            .apply(lambda x: len(set(x)))
            .reset_index(level=2, drop=True)
            .reset_index(name="_daily_cts_wo_lag")
            .drop_duplicates([level, "date"], keep="last")
        )
        df = df.merge(daily_cts_wo_lag, on=[level, "date"], how="left")

    df[col_name] = df[col_name].bfill()
    df["_comb_col"] = np.where(
        df[col_name].isnull(), df[col_name], df["_daily_cts_wo_lag"],
    )
    df["_comb_col"] = df._comb_col.ffill()
    df[col_name] = df[col_name].fillna(df._comb_col)
    df.drop(["_daily_cts_wo_lag", "_comb_col"], axis=1, inplace=True)

    return _downcast(df)


def _expanding_max(idx, df, levels):
    """Helper function used in creating column with days elapsed since day with
    maximum quantity sold (before current day).

    Parameters:
    -----------
    idx : int or tuple
        Value(s) defining group in group-by
    df : pandas DataFrame
        Dataframe in which to create new column
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    numpy array (1-d)
    """
    # if isinstance(idx, int):
    #     df.index = pd.Index([idx] * len(df)).set_names(
    #         [level + "_id" for level in levels]
    #     )
    # elif isinstance(idx, tuple):
    #     df.index = pd.MultiIndex.from_tuples([idx] * len(df)).set_names(
    #         [level + "_id" for level in levels]
    #     )
    exp = (
        df.set_index("date")[f"{'_'.join(levels)}_qty_sold_day"]
        # df.set_index("date", append=True)[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .max()
    )
    # return (
    #     exp.groupby(exp)
    #     .transform("idxmax")
    #     .apply(lambda x: pd.to_datetime(x[len(levels)]))
    #     .values
    # )
    m = {k: v for k, v in zip(*np.unique(exp.values, return_index=True))}
    idx_of_first_occ = np.array([m[i] for i in exp.values], dtype='uint32')
    return np.array([exp.index[i] for i in idx_of_first_occ], dtype='datetime64[D]')


@Timer(logger=logging.info)
def days_since_max_qty_sold(df, levels):
    """Create column with days elapsed since day with maximum quantity sold
    (before current day).

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new columns (dataframe will be modified inplace)
    levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item'])

    Returns:
    --------
    df : pandas DataFrame
        Updated dataframe

    Notes:
    ------
    Function assumes that the earliest (first) row for each value of levels has a
    non-zero quantity sold value.
    """
    results = []
    for i, (g, grp) in enumerate(df.groupby([level + "_id" for level in levels])):
        if i % 25 == 0:
            gc.collect()
        results.append(_expanding_max(g, grp, levels))
        # can make results a numpy array (results = np.empty((0,), dtype=np.___))
        # or results = np.array([], dtype=np.int8)
        # then append to that array (results = np.append(results, ___))
        # not sure if this would be beneficial, but it would create a flat
        # array from start, instead of a list of arrays

    df["date_of_max_qty"] = np.concatenate(results, axis=0)
    del results

    df["date_of_max_qty"] = df.groupby(
        [level + "_id" for level in levels]
    ).date_of_max_qty.shift()

    df.loc[df.date_of_max_qty.isnull(), "date_of_max_qty"] = df.date

    df["days_since_max_qty_sold"] = (df.date - df.date_of_max_qty).dt.days
    df.drop("date_of_max_qty", axis=1, inplace=True)
    return _downcast(df)


@Timer(logger=logging.info)
def build_item_date_lvl_features(return_df=False, to_sql=False, test_run=False):
    """Build dataframe of item-date-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of item_id and date and columns containing item-date-level features
    """
    # check if cleaned sales file already exists
    if test_run:
        input_csv = "sales_cleaned_test_run.csv"
    else:
        input_csv = "sales_cleaned.csv"
    cleaned_sales_file = Path(input_csv)
    if cleaned_sales_file.is_file():
        sales = pd.read_csv(input_csv)
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(return_df=True, test_run=test_run)

    # Quantity Sold

    # create column with quantity sold by item-date
    item_date_level_features = (
        sales.groupby(["item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "item_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    item_date_level_features = (
        item_date_level_features.pipe(add_zero_qty_rows, ["item"])
        .pipe(add_zero_qty_after_last_date_rows, ["item"])
        .pipe(prev_nonzero_qty_sold, ["item"])
        .pipe(days_elapsed_since_prev_sale, ["item"])
        .pipe(days_elapsed_since_first_sale, ["item"])
        .pipe(first_week_month_of_sale, ["item"])
        .pipe(num_of_sale_dts_in_prev_x_days, ["item"])
        .pipe(rolling_7d_qty_stats, ["item"])
        .pipe(expanding_cv2_of_qty, ["item"])
        .pipe(expanding_avg_demand_int, ["item"])
        .pipe(expanding_qty_sold_stats, ["item"])
        .pipe(qty_sold_x_days_before, ["item"])
        .pipe(expanding_time_bw_sales_stats, ["item"])
        .pipe(diff_bw_last_and_sec_to_last_qty, ["item"])
        .pipe(num_of_unique_opp_values, sales, "item_id")
        .pipe(days_since_max_qty_sold, ["item"])
    )

    # Demand for Category in Last Week (Quantity Sold, Count of Unique Items Sold, Quantity Sold per Item)

    # Add item_category_id column
    item_date_level_features = item_date_level_features.merge(
        items_df[["item_id", "item_category_id"]], on="item_id", how="left"
    )

    # create dataframe with daily totals of quantity sold for each category
    cat_date_total_qty = (
        item_date_level_features[["date", "item_category_id", "item_qty_sold_day"]]
        .groupby("item_category_id")
        .apply(lambda x: x.resample("D", on="date").item_qty_sold_day.sum())
    ).reset_index(name="cat_qty_sold_day")

    # calculate rolling weekly sum of quantity sold for each category, excluding current date
    cat_date_total_qty["cat_qty_sold_last_7d"] = cat_date_total_qty.groupby(
        "item_category_id"
    )["cat_qty_sold_day"].apply(lambda x: x.rolling(7, 1).sum().shift().fillna(0))

    # merge rolling weekly category quantity totals onto item-date dataset
    item_date_level_features = item_date_level_features.merge(
        cat_date_total_qty[["item_category_id", "date", "cat_qty_sold_last_7d"]],
        on=["item_category_id", "date"],
        how="left",
    )

    cat_date_item_cts = (
        item_date_level_features[
            ["item_category_id", "date", "item_id", "item_qty_sold_day"]
        ]
        .set_index("date")
        .sort_values(by=["item_category_id", "date"])
    )

    cat_date_item_cts["item_id_mdfd"] = np.where(
        cat_date_item_cts.item_qty_sold_day <= 0, -999, cat_date_item_cts.item_id
    )
    cat_date_item_cts.drop("item_qty_sold_day", axis=1, inplace=True)

    cat_date_item_cts = (
        cat_date_item_cts.groupby("item_category_id")["item_id_mdfd"]
        .rolling("7D")
        .apply(
            lambda x: np.unique(np.ma.MaskedArray(x, mask=(np.array(x < 0)))).count(),
            raw=True
            # engine="numba",
            # engine_kwargs={"nopython": False},
        )
        .reset_index(name="cat_unique_items_sold_last_7d")
        .drop_duplicates(["item_category_id", "date"], keep="last")
    )

    cat_date_item_cts = (
        cat_date_item_cts.groupby("item_category_id")
        .apply(lambda x: x.set_index("date").resample("D").asfreq())
        .drop("item_category_id", axis=1)
        .reset_index()
    )
    cat_date_item_cts["cat_unique_items_sold_last_7d"] = cat_date_item_cts.groupby(
        "item_category_id"
    ).cat_unique_items_sold_last_7d.apply(lambda x: x.ffill().shift().fillna(0))

    item_date_level_features = item_date_level_features.merge(
        cat_date_item_cts[
            ["item_category_id", "date", "cat_unique_items_sold_last_7d"]
        ],
        on=["item_category_id", "date"],
        how="left",
    )

    # add column with quantity sold per item in category in the last week
    item_date_level_features["cat_qty_sold_per_item_last_7d"] = (
        item_date_level_features["cat_qty_sold_last_7d"]
        .div(item_date_level_features["cat_unique_items_sold_last_7d"])
        .replace([np.inf, np.nan], 0)
    )

    # Presence of Spikes in Quantity Sold

    # column indicating whether item had a spike in quantity sold before current day
    res = (
        item_date_level_features.groupby("item_id")["item_qty_sold_day"]
        .expanding()
        .apply(
            _spike_check,
            raw=True
            # _spike_check, raw=True, engine="numba", engine_kwargs={"nopython": False}
        )
        .values
    )
    item_date_level_features["item_had_spike_before_day"] = res.astype(bool).astype(
        np.int8
    )

    # column with count of spikes in quantity sold before current day
    item_date_level_features["item_n_spikes_before_day"] = res.astype(np.int8)

    item_date_level_features = _downcast(item_date_level_features)
    item_date_level_features = _add_col_prefix(item_date_level_features, "id_")

    logging.info(
        f"Item-date-level dataframe has {item_date_level_features.shape[0]} rows and "
        f"{item_date_level_features.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Item-date-level dataframe has the following columns: "
        f"{nl}{nl.join(item_date_level_features.columns.to_list())}"
    )
    logging.info(
        f"Item-date dataframe has the following data types: "
        f"{nl}{item_date_level_features.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = item_date_level_features.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Item-date-level dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    if to_sql:
        start_instance()
        item_date_level_features.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(item_date_level_features)
        write_df_to_sql(item_date_level_features, "item_dates", dtypes_dict)
        stop_instance()

    if return_df:
        return item_date_level_features


# SHOP-DATE-LEVEL FEATURES
@Timer(logger=logging.info)
def build_shop_date_lvl_features(return_df=False, to_sql=False, test_run=False):
    """Build dataframe of shop-date-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    shop_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id and date and columns containing shop-date-level features
    """
    # check if cleaned sales file already exists
    if test_run:
        input_csv = "sales_cleaned_test_run.csv"
    else:
        input_csv = "sales_cleaned.csv"
    cleaned_sales_file = Path(input_csv)
    if cleaned_sales_file.is_file():
        sales = pd.read_csv(input_csv)
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(return_df=True, test_run=test_run)

    # Quantity Sold by Shop-Date

    # create column with quantity sold by shop-date
    shop_date_level_features = (
        sales.groupby(["shop_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    shop_date_level_features = (
        shop_date_level_features.pipe(add_zero_qty_rows, ["shop"])
        .pipe(add_zero_qty_after_last_date_rows, ["shop"])
        .pipe(prev_nonzero_qty_sold, ["shop"])
        .pipe(days_elapsed_since_prev_sale, ["shop"])
        .pipe(days_elapsed_since_first_sale, ["shop"])
        .pipe(first_week_month_of_sale, ["shop"])
        .pipe(num_of_sale_dts_in_prev_x_days, ["shop"])
        .pipe(rolling_7d_qty_stats, ["shop"])
        .pipe(expanding_qty_sold_stats, ["shop"])
        .pipe(qty_sold_x_days_before, ["shop"])
        .pipe(expanding_time_bw_sales_stats, ["shop"])
        .pipe(diff_bw_last_and_sec_to_last_qty, ["shop"])
        .pipe(num_of_unique_opp_values, sales, "shop_id")
    )

    # Number of Unique Categories of Items Sold at Shops Prior to Current Day

    sales_sorted = sales[["shop_id", "item_id", "date"]].sort_values(
        by=["shop_id", "date"], ignore_index=True
    )

    # add item_category_id column
    sales_sorted = sales_sorted.merge(
        items_df[["item_id", "item_category_id"]], on="item_id", how="left"
    )

    shop_dates_w_item_cat_cts = sales_sorted[["item_category_id", "date"]]
    shop_dates_w_item_cat_cts.index = [
        sales_sorted.shop_id,
        sales_sorted.groupby("shop_id").cumcount().rename("sidx"),
    ]

    shop_dates_w_item_cat_cts = (
        shop_dates_w_item_cat_cts.groupby(level="shop_id")
        .apply(_lag_merge_asof, "item_category_id", lag=1)
        .reset_index("shop_id")
        .reset_index(drop=True)
        .rename(
            columns={
                "num_unique_values_prior_to_day": "num_unique_item_cats_prior_to_day"
            }
        )
    )
    shop_dates_w_item_cat_cts.drop_duplicates(subset=["shop_id", "date"], inplace=True)
    shop_dates_w_item_cat_cts.num_unique_item_cats_prior_to_day.fillna(0, inplace=True)
    shop_dates_w_item_cat_cts.drop("item_category_id", axis=1, inplace=True)

    shop_date_level_features = shop_date_level_features.merge(
        shop_dates_w_item_cat_cts, on=["shop_id", "date"], how="left"
    )

    # fill null values on days when no sale was made
    daily_cts_wo_lag = (
        sales_sorted.groupby(["shop_id", "date"])["item_category_id"]
        .expanding()
        .apply(lambda x: len(set(x)))
        .reset_index(level=2, drop=True)
        .reset_index(name="_daily_cts_wo_lag")
        .drop_duplicates(["shop_id", "date"], keep="last")
    )
    shop_date_level_features = shop_date_level_features.merge(
        daily_cts_wo_lag, on=["shop_id", "date"], how="left"
    )
    shop_date_level_features[
        "num_unique_item_cats_prior_to_day"
    ] = shop_date_level_features["num_unique_item_cats_prior_to_day"].bfill()
    shop_date_level_features["_comb_col"] = np.where(
        shop_date_level_features["num_unique_item_cats_prior_to_day"].isnull(),
        shop_date_level_features["num_unique_item_cats_prior_to_day"],
        shop_date_level_features["_daily_cts_wo_lag"],
    )
    shop_date_level_features["_comb_col"] = shop_date_level_features._comb_col.ffill()
    shop_date_level_features[
        "num_unique_item_cats_prior_to_day"
    ] = shop_date_level_features["num_unique_item_cats_prior_to_day"].fillna(
        shop_date_level_features._comb_col
    )
    shop_date_level_features.drop(
        ["_daily_cts_wo_lag", "_comb_col"], axis=1, inplace=True
    )

    shop_date_level_features = _downcast(shop_date_level_features)
    shop_date_level_features = _add_col_prefix(shop_date_level_features, "sd_")

    logging.info(
        f"Shop-date-level dataframe has {shop_date_level_features.shape[0]} rows and "
        f"{shop_date_level_features.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Shop-date-level dataframe has the following columns: "
        f"{nl}{nl.join(shop_date_level_features.columns.to_list())}"
    )
    logging.info(
        f"Shop-date dataframe has the following data types: "
        f"{nl}{shop_date_level_features.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = shop_date_level_features.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Shop-date-level dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    if to_sql:
        start_instance()
        shop_date_level_features.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(shop_date_level_features)
        write_df_to_sql(shop_date_level_features, "shop_dates", dtypes_dict)
        stop_instance()

    if return_df:
        return shop_date_level_features


def _mad(data, axis=None):
    """Calculate mean absolute deviation of array.

    Parameters:
    -----------
    data : array-like
        Array of values on which to run the function
    axis : int (or None)
        Array axis on which to run the function

    Returns:
    --------
    Expression for mean absolute deviation
    """
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def _daily_qty_sum(idx, df, grp_levels):
    """Helper function used in computing daily quantity totals by shop-item category.

    Parameters:
    -----------
    idx : index to assign to output dataframe (will be named same as values in
        grp_levels list)
    df : pandas DataFrame
        Input dataframe
    grp_levels : list of strings
        List of levels by which to group values (e.g., ['shop', 'item_category_id'])

    Returns:
    --------
    pandas DataFrame
    """
    out_df = (
        df.resample("D", on="date")
        .shop_item_qty_sold_day.sum()
        .reset_index(name="shop_cat_qty_sold_day")
    )
    out_df.index = pd.MultiIndex.from_tuples([idx] * len(out_df)).set_names(grp_levels)
    return out_df.reset_index()


def _roll_weekly_sum(df):
    """Helper function used in creating column with rolling weekly sum of
    quantity sold by item category at each shop.

    Parameters:
    -----------
    df : pandas Dataframe
        Input dataframe

    Returns:
    --------
    pandas DataFrame
    """
    df["shop_cat_qty_sold_last_7d"] = (
        df["shop_cat_qty_sold_day"].rolling(7, 1).sum().shift().fillna(0).astype('int16')
    )
    return df


def _expand_sale_flag(df):
    """Helper function used in creating column with binary flag indicating
    whether shop sold an item in same item category before current day.

    Parameters:
    -----------
    df : pandas Dataframe
        Input dataframe

    Returns:
    --------
    pandas DataFrame
    """
    df["cat_sold_at_shop_before_day_flag"] = (
        df["shop_cat_qty_sold_day"]
        .expanding()
        .sum()
        .shift()
        .fillna(0)
        .astype(bool)
        .astype(np.int8)
    )
    return df


def _shift_ffill_fillna(df):
    """Helper function used in creating columns where values are shifted and
    null values are filled within groups.

    Parameters:
    -----------
    df : pandas Dataframe
        Input dataframe

    Returns:
    --------
    pandas DataFrame
    """
    df["coef_var_price"] = df["coef_var_price"].shift().ffill().fillna(0).astype('float32')
    df["qty_mean_abs_dev"] = df["qty_mean_abs_dev"].shift().ffill().fillna(0).astype('float32')
    df["qty_median_abs_dev"] = df["qty_median_abs_dev"].shift().ffill().fillna(0).astype('float32')
    return df


# SHOP-ITEM-DATE-LEVEL FEATURES
@Timer(logger=logging.info)
def build_shop_item_date_lvl_features(return_df=False, to_sql=False, test_run=False):
    """Build dataframe of shop-item-date-level features.

    Parameters:
    -----------
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)
    test_run : bool
        Run code only on last month of available data (True) or not (False)

    Returns:
    --------
    shop_item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id,
        item_id, and date and columns containing shop-item-date-level features
    """
    # check if cleaned sales file already exists
    if test_run:
        input_csv = "sales_cleaned_test_run.csv"
    else:
        input_csv = "sales_cleaned.csv"
    cleaned_sales_file = Path(input_csv)
    if cleaned_sales_file.is_file():
        sales = pd.read_csv(input_csv)
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(return_df=True, test_run=test_run)

    # Total Quantity Sold by Shop-Item-Date

    # # create column with quantity sold by shop-item-date
    shop_item_date_level_features = _downcast(
        sales.groupby(["shop_id", "item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_item_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    shop_item_date_level_features = (shop_item_date_level_features
        .pipe(add_zero_qty_rows, ["shop", "item"])
        .pipe(add_zero_qty_after_last_date_rows, ["shop", "item"])
        .pipe(prev_nonzero_qty_sold, ["shop", "item"])
        .pipe(days_elapsed_since_prev_sale, ["shop", "item"])
        .pipe(days_elapsed_since_first_sale, ["shop", "item"])
        .pipe(first_week_month_of_sale, ["shop", "item"])
        .pipe(num_of_sale_dts_in_prev_x_days, ["shop", "item"], to_sql=to_sql)
        .pipe(rolling_7d_qty_stats, ["shop", "item"], to_sql=to_sql)
        .pipe(expanding_cv2_of_qty, ["shop", "item"], to_sql=to_sql)
        .pipe(expanding_avg_demand_int, ["shop", "item"])
        .pipe(expanding_qty_sold_stats, ["shop", "item"], to_sql=to_sql)
        .pipe(qty_sold_x_days_before, ["shop", "item"])
        .pipe(expanding_time_bw_sales_stats, ["shop", "item"], to_sql=to_sql)
        .pipe(diff_bw_last_and_sec_to_last_qty, ["shop", "item"])
        .pipe(days_since_max_qty_sold, ["shop", "item"])
    )

    # SAVE MAIN DF TO SQL TABLE AND DELETE OBJECT
    shop_item_date_level_features = _downcast(shop_item_date_level_features)
    shop_item_date_level_features.rename(columns={"date": "sale_date"}, inplace=True)
    dtypes_dict = _map_to_sql_dtypes(shop_item_date_level_features)

    input_json = "master_pd_types.json"
    types_json = Path(input_json)
    if types_json.is_file():
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)
        if "items" not in master_pd_types:
            build_item_lvl_features()
            del master_pd_types
            with open(input_json, "r") as fp:
                master_pd_types = json.load(fp)
    else:
        build_item_lvl_features()
        with open(input_json, "r") as fp:
            master_pd_types = json.load(fp)
    master_pd_types.update(
        {"sid": shop_item_date_level_features.dtypes.map(str).to_dict()}
    )

    start_instance()
    write_df_to_sql(shop_item_date_level_features, "df_temp", dtypes_dict)
    del shop_item_date_level_features

    # Expanding Coefficient of Variation of item price (across all dates for shop-item before current date)

    # expanding coefficient of variation of price across dates with a sale for each shop-item
    coefs_var = _downcast(
        sales.groupby(["shop_id", "item_id"])["item_price"]
        .expanding()
        .agg(variation)
        .reset_index(name="coef_var_price")
        .drop("level_2", axis=1)
    )

    # add date column
    coefs_var = pd.concat([coefs_var, sales[["date"]]], axis=1)

    # SAVE COEFS_VAR TO SQL TABLE AND DELETE OBJECT
    coefs_var = _downcast(coefs_var)
    coefs_var.rename(columns={"date": "sale_date"}, inplace=True)
    dtypes_dict = _map_to_sql_dtypes(coefs_var)
    master_pd_types.update({"coefs_var": coefs_var.dtypes.map(str).to_dict()})
    write_df_to_sql(coefs_var, "coefs_var", dtypes_dict)
    del coefs_var

    # Expanding Mean Absolute Deviation of Quantity Sold (across all shop-items before current date)

    # expanding Mean Absolute Deviation of Quantity Sold across dates with a sale for each shop-item
    qty_mads = _downcast(
        sales.groupby(["shop_id", "item_id"])["item_cnt_day"]
        .expanding()
        .agg(_mad)
        .reset_index(name="qty_mean_abs_dev")
        .drop("level_2", axis=1)
    )

    # add date column
    qty_mads = pd.concat([qty_mads, sales[["date"]]], axis=1)

    # SAVE QTY_MADS TO SQL TABLE AND DELETE OBJECT
    qty_mads = _downcast(qty_mads)
    qty_mads.rename(columns={"date": "sale_date"}, inplace=True)
    dtypes_dict = _map_to_sql_dtypes(qty_mads)
    master_pd_types.update({"qty_mads": qty_mads.dtypes.map(str).to_dict()})
    write_df_to_sql(qty_mads, "qty_mads", dtypes_dict)
    del qty_mads

    # Expanding Median Absolute Deviation of Quantity Sold (across all shop-items before current date)

    # expanding Median Absolute Deviation of Quantity Sold across dates with a sale for each shop-item
    qty_median_ads = _downcast(
        sales.groupby(["shop_id", "item_id"])["item_cnt_day"]
        .expanding()
        .agg(median_absolute_deviation)
        .reset_index(name="qty_median_abs_dev")
        .drop("level_2", axis=1)
    )

    # add date column
    qty_median_ads = pd.concat([qty_median_ads, sales[["date"]]], axis=1)

    # SAVE QTY_MEDIAN_ADS TO SQL TABLE AND DELETE OBJECT
    qty_median_ads = _downcast(qty_median_ads)
    qty_median_ads.rename(columns={"date": "sale_date"}, inplace=True)
    dtypes_dict = _map_to_sql_dtypes(qty_median_ads)
    master_pd_types.update({"qty_median_ads": qty_median_ads.dtypes.map(str).to_dict()})
    write_df_to_sql(qty_median_ads, "qty_median_ads", dtypes_dict)
    del qty_median_ads

    with open("master_pd_types.json", "w") as fp:
        json.dump(master_pd_types, fp)

    # PERFORM JOIN OF ALL TABLES INSIDE RDS AND SAVE TO NEW DF
    db_table_name = "sid_mult_left_join"
    sql_str = (
        "CREATE TABLE {0} AS SELECT a.*, b.coef_var_price, c.qty_mean_abs_dev, d.qty_median_abs_dev, e.i_item_category_id "
        "FROM df_temp a "
        "LEFT JOIN coefs_var b "
        "ON a.shop_id = b.shop_id AND a.item_id = b.item_id AND a.sale_date = b.sale_date "
        "LEFT JOIN qty_mads c "
        "ON a.shop_id = c.shop_id AND a.item_id = c.item_id AND a.sale_date = c.sale_date "
        "LEFT JOIN qty_median_ads d "
        "ON a.shop_id = d.shop_id AND a.item_id = d.item_id AND a.sale_date = d.sale_date "
        "LEFT JOIN items e "
        "ON a.item_id = e.item_id "
        "ORDER BY shop_id, item_id, sale_date;"
    )
    sql = SQL(sql_str).format(Identifier(db_table_name))

    # Write query results to new SQL table
    params = {"db_table": db_table_name}
    create_db_table_from_query(sql, params)
    cast_dict = {
                    **master_pd_types["coefs_var"],
                    **master_pd_types["qty_mads"],
                    **master_pd_types["qty_median_ads"],
                    **master_pd_types["items"],
                    **master_pd_types["sid"],
                }
    shop_item_date_level_features = df_from_sql_table(db_table_name, cast_dict, date_list=["sale_date"])
    # If df is successfully created, delete the big db tables from which it was created
    if shop_item_date_level_features is not None:
        drop_tables(["df_temp", "coefs_var", "qty_mads", "qty_median_ads"] + [db_table_name])

    # shop_item_date_level_features = df_from_sql_query(
    #     sql,
    #     {
    #         **master_pd_types["coefs_var"],
    #         **master_pd_types["qty_mads"],
    #         **master_pd_types["qty_median_ads"],
    #         **master_pd_types["items"],
    #         **master_pd_types["sid"],
    #     },
    #     date_list=["sale_date"],
    #     delete_tables=["df_temp", "coefs_var", "qty_mads", "qty_median_ads"],
    # )
    shop_item_date_level_features.rename(
        columns={"sale_date": "date", "i_item_category_id": "item_category_id"},
        inplace=True,
    )

    results = []
    for i, (g, grp) in enumerate(
        shop_item_date_level_features.groupby(["shop_id", "item_id"])
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_shift_ffill_fillna(grp))
    del shop_item_date_level_features
    shop_item_date_level_features = pd.concat(results, ignore_index=True)
    del results

    # create dataframe with daily totals of quantity sold for each category at each shop
    grp_levels = ["shop_id", "item_category_id"]

    results = []
    for i, (g, grp) in enumerate(
        shop_item_date_level_features[
            grp_levels + ["date"] + ["shop_item_qty_sold_day"]
        ].groupby(grp_levels)
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_daily_qty_sum(g, grp, grp_levels))
    shop_cat_date_total_qty = _downcast(pd.concat(results, ignore_index=True))
    del results

    # calculate rolling weekly sum of quantity sold for each category at each shop, excluding current date
    results = []
    for i, (g, grp) in enumerate(
        shop_cat_date_total_qty[
            grp_levels + ["date"] + ["shop_cat_qty_sold_day"]
        ].groupby(grp_levels)
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_roll_weekly_sum(grp))
    shop_cat_date_total_qty = _downcast(pd.concat(results, ignore_index=True))
    del results

    results = []
    for i, (g, grp) in enumerate(
        shop_cat_date_total_qty[
            grp_levels + ["date"] + ["shop_cat_qty_sold_day"]
        ].groupby(grp_levels)
    ):
        if i % 100 == 0:
            gc.collect()
        results.append(_expand_sale_flag(grp))
    shop_cat_date_total_qty = _downcast(pd.concat(results, ignore_index=True))
    del results

    # export shop-category-date DF to separate SQL table and delete object
    if to_sql:
        # shop_cat_date_total_qty = _downcast(shop_cat_date_total_qty)
        shop_cat_date_total_qty = _add_col_prefix(shop_cat_date_total_qty, "sid_")
        logging.info(
            f"Shop-category-date dataframe has {shop_cat_date_total_qty.shape[0]} "
            f"rows and {shop_cat_date_total_qty.shape[1]} columns."
        )
        nl = "\n" + " " * 55
        logging.info(
            f"Shop-category-date dataframe has the following columns: "
            f"{nl}{nl.join(shop_cat_date_total_qty.columns.to_list())}"
        )
        logging.info(
            f"Shop-category-date dataframe has the following data types: "
            f"{nl}{shop_cat_date_total_qty.dtypes.to_string().replace(nl[:1],nl)}"
        )
        miss_vls = shop_cat_date_total_qty.isnull().sum()
        miss_vls = miss_vls[miss_vls > 0].index.to_list()
        if miss_vls:
            logging.warning(
                f"Shop-category-date dataframe has the following columns with "
                f"null values: {nl}{nl.join(miss_vls)}"
            )
        shop_cat_date_total_qty.rename(columns={"date": "sale_date"}, inplace=True)
        dtypes_dict = _map_to_sql_dtypes(shop_cat_date_total_qty)
        write_df_to_sql(shop_cat_date_total_qty, "shop_cat_dates", dtypes_dict)
        del shop_cat_date_total_qty

    # shop_item_date_level_features = _downcast(shop_item_date_level_features)
    shop_item_date_level_features = _add_col_prefix(
        shop_item_date_level_features, "sid_"
    )

    logging.info(
        f"Shop-item-date dataframe has {shop_item_date_level_features.shape[0]} "
        f"rows and {shop_item_date_level_features.shape[1]} columns."
    )
    nl = "\n" + " " * 55
    logging.info(
        f"Shop-item-date dataframe has the following columns: "
        f"{nl}{nl.join(shop_item_date_level_features.columns.to_list())}"
    )
    logging.info(
        f"Shop-item-date dataframe has the following data types: "
        f"{nl}{shop_item_date_level_features.dtypes.to_string().replace(nl[:1],nl)}"
    )
    miss_vls = shop_item_date_level_features.isnull().sum()
    miss_vls = miss_vls[miss_vls > 0].index.to_list()
    if miss_vls:
        logging.warning(
            f"Shop-item-date dataframe has the following columns with "
            f"null values: {nl}{nl.join(miss_vls)}"
        )

    if to_sql:
        start_instance()
        shop_item_date_level_features.rename(
            columns={"date": "sale_date"}, inplace=True
        )
        dtypes_dict = _map_to_sql_dtypes(shop_item_date_level_features)
        write_df_to_sql(shop_item_date_level_features, "shop_item_dates", dtypes_dict)
        stop_instance()

    if return_df:
        return shop_item_date_level_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'clean', 'shops', 'items', 'dates', 'item-dates', 'shop-dates' or 'shop-item-dates'",
    )
    parser.add_argument(
        "--send_to_sql",
        default=False,
        action="store_true",
        help="write DF to SQL (if included) or not (if not included)",
    )
    parser.add_argument(
        "--test_run",
        default=False,
        action="store_true",
        help="run code only on last month of data (if included) or on all data (if not)",
    )

    args = parser.parse_args()

    if args.command not in [
        "clean",
        "shops",
        "items",
        "dates",
        "item-dates",
        "shop-dates",
        "shop-item-dates",
    ]:
        print(
            "'{}' is not recognized. "
            "Use 'clean', 'shops', 'items', 'dates', 'item-dates' "
            "'shop-dates' or 'shop-item-dates'".format(args.command)
        )

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("logs")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    log_fname = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_{args.command}.log"
    log_path = log_dir.joinpath(log_fname)

    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format=fmt,
        datefmt=datefmt,
        filename=log_path,
    )

    # statements to suppress irrelevant logging by boto3-related libraries
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # Check if code is being run on EC2 instance (vs locally)
    my_user = os.environ.get("USER")
    is_aws = True if "ec2" in my_user else False
    # Log EC2 instance name and type metadata
    if is_aws:
        instance_metadata = dict()
        instance_metadata['EC2 instance ID'] = ec2_metadata.instance_id
        instance_metadata['EC2 instance type'] = ec2_metadata.instance_type
        instance_metadata['EC2 instance public hostname'] = ec2_metadata.public_hostname

        f = lambda x: ": ".join(x)
        r = list(map(f, list(instance_metadata.items())))
        nl = "\n" + " " * 55
        logging.info(
            f"Script is running on EC2 instance with the following metadata: "
            f"{nl}{nl.join(r)}"
        )
    else:
        logging.info("Script is running on local machine, not on EC2 instance.")

    logging.info(f"The Python version is {platform.python_version()}.")
    logging.info(f"The pandas version is {pd.__version__}.")
    logging.info(f"The numpy version is {np.__version__}.")
    logging.info(f"The SQLAlchemy version is {sqlalchemy.__version__}")

    # Call clean_sales_data() if need to run that code in standalone fashion.
    # If calling other functions that rely on clean sales data, check if clean
    # data file exists and use it if it does or call clean_sales_data() if
    # it does not.

    # Change value of start of training period if doing a test run
    global FIRST_DAY_OF_TRAIN_PRD
    if args.test_run:
        FIRST_DAY_OF_TRAIN_PRD = (2015, 10, 1)

    if args.command == "clean":
        clean_sales_data(to_sql=args.send_to_sql, test_run=args.test_run)
    elif args.command == "shops":
        build_shop_lvl_features(to_sql=args.send_to_sql)
    elif args.command == "items":
        build_item_lvl_features(to_sql=args.send_to_sql, test_run=args.test_run)
    elif args.command == "dates":
        build_date_lvl_features(to_sql=args.send_to_sql, test_run=args.test_run)
    elif args.command == "item-dates":
        build_item_date_lvl_features(to_sql=args.send_to_sql, test_run=args.test_run)
    elif args.command == "shop-dates":
        build_shop_date_lvl_features(to_sql=args.send_to_sql, test_run=args.test_run)
    elif args.command == "shop-item-dates":
        build_shop_item_date_lvl_features(
            to_sql=args.send_to_sql, test_run=args.test_run
        )

    # copy log file to S3 bucket
    upload_file(f"./logs/{log_fname}", "my-ec2-logs", log_fname)


if __name__ == "__main__":
    main()
