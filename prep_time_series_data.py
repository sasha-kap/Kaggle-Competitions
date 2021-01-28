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
"""

# Standard library imports
import argparse
import datetime
import logging
from pathlib import Path
import platform
import warnings

# Third-party library imports
import bottleneck as bn
import numpy as np
import pandas as pd
import scipy
from scipy.stats import median_absolute_deviation, variation
from tqdm import tqdm

# Local imports
from constants import (
    FIRST_DAY_OF_TRAIN_PRD,
    LAST_DAY_OF_TRAIN_PRD,
    FIRST_DAY_OF_TEST_PRD,
    PUBLIC_HOLIDAYS,
    PUBLIC_HOLIDAY_DTS,
    OLYMPICS2014,
    WORLDCUP2014,
    CITY_POP,
)
from timer import Timer
from write_df_to_sql_table import psql_insert_copy, write_df_to_sql

warnings.filterwarnings("ignore")

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
    df_to_use = df.copy()

    return df_to_use.assign(
        **{col: transform_fn(df_to_use[col]) for col in condition(df_to_use)}
    )


def _all_float_to_int(df):
    df_to_use = df.copy()
    transform_fn = _float_to_int
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)

    return _multi_assign(df_to_use, transform_fn, condition)


def _downcast_all(df, target_type, initial_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if initial_type is None:
        initial_type = target_type

    df_to_use = df.copy()

    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)

    condition = lambda x: list(
        x.select_dtypes(include=[initial_type], exclude=["uint32"]).columns
    )

    return _multi_assign(df_to_use, transform_fn, condition)

@Timer(logger = logging.info)
def _downcast(df_in):
    return (
        df_in.pipe(_all_float_to_int)
        .pipe(_downcast_all, "float")
        .pipe(_downcast_all, "integer")
        .pipe(_downcast_all, target_type="unsigned", initial_type="integer")
    )


# PERFORM INITIAL DATA CLEANING
@Timer(logger = logging.info)
def clean_sales_data(sales, return_df=False, to_sql=False):
    """Perform initial data cleaning of the train shop-item-date data.

    Parameters:
    -----------
    sales : pandas Dataframe
        Original dataframe at roughly shop-item-date level for the train period
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    sales : pandas dataframe
        Cleaned-up version of shop-item-date train dataset
    """

    # convert the date column from string to datetime type
    sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y"))

    # Drop duplicate rows
    sales.drop_duplicates(inplace=True)

    # Identify duplicate rows by shop-item-date
    dupes = sales.loc[
        sales.duplicated(subset=["shop_id", "item_id", "date"], keep=False), :
    ]

    # Remove two pairs of shop-item-dates with multiple quantities when one quantity was negative
    dupes = dupes[
        ~((dupes.shop_id == 38) & (dupes.item_id == 15702))
        & ~((dupes.shop_id == 5) & (dupes.item_id == 21619))
    ]

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
    sales.to_csv("sales_cleaned.csv", index=False)

    sales.rename(columns={'date':'sale_date'}, inplace=True)

    if to_sql:
        write_df_to_sql(sales, "sales_cleaned")

    if return_df:
        return sales

@Timer(logger = logging.info)
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


@Timer(logger = logging.info)
def build_shop_lvl_features(shops_df, return_df=False, to_sql=False):
    """Build dataframe of shop-level features.

    Parameters:
    -----------
    shops_df : pandas DataFrame
        Existing dataframe with shop_id to shop_name and city mapping
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
    shops_df = shops_df[~(shops_df.shop_id.isin([9, 20]))]

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

    shops_df = _downcast(shops_df)
    shops_df = _add_col_prefix(shops_df, "s_")

    if to_sql:
        write_df_to_sql(shops_df, "shops")

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


@Timer(logger = logging.info)
def build_item_lvl_features(items_df, categories_df, return_df=False, to_sql=False):
    """Build dataframe of item-level features.

    Parameters:
    -----------
    items_df : pandas DataFrame
        Existing dataframe with item_id to item_category_id mapping
    categories_df : pandas DataFrame
        Existing dataframe with item_category_id to item_category_name mapping
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    item_level_features : pandas dataframe
        Dataframe with each row representing an item and columns containing item-level features
    """
    # check if cleaned sales file already exists
    cleaned_sales_file = Path("sales_cleaned.csv")
    if cleaned_sales_file.is_file():
        sales = pd.read_csv("sales_cleaned.csv")
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(sales, return_df=True)

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

    if to_sql:
        write_df_to_sql(item_level_features, "items")

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

@Timer(logger = logging.info)
def build_date_lvl_features(macro_df, ps4games, sales, return_df=False, to_sql=False):
    """Build dataframe of date-level features.

    Parameters:
    -----------
    macro_df : pandas DataFrame
        Existing dataframe with columns containing daily macroeconomic indicators
    ps4games : pandas DataFrame
        Existing dataframe with PS4 game release dates
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    date_level_features : pandas dataframe
        Dataframe with each row representing a date and columns containing date-level features
    """

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
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
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

    # convert the date column in macro_df from string to datetime type
    macro_df["date"] = pd.to_datetime(macro_df.timestamp)

    # subset macro_df dataset to relevant period
    macro_df_2013_2015 = macro_df[
        (macro_df.date >= datetime.datetime(*FIRST_DAY_OF_TRAIN_PRD))
        & (macro_df.date <= datetime.datetime(*LAST_DAY_OF_TRAIN_PRD))
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
    ps4games[["Release date JP", "Release date EU", "Release date NA"]] = ps4games[
        ["Release date JP", "Release date EU", "Release date NA"]
    ].apply(pd.to_datetime, errors="coerce")

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
        ] = date_level_features.day_total_qty_sold.shift(shift_val)

    # create column for 1-day lagged brent price (based on the results of cross-correlation analysis)
    date_level_features["brent_1day_lag"] = date_level_features.brent.shift(1)

    date_level_features = _downcast(date_level_features)
    date_level_features = _add_col_prefix(date_level_features, "d_")

    date_level_features.rename(columns={'date':'sale_date'}, inplace=True)

    if to_sql:
        write_df_to_sql(date_level_features, "dates")

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
        Dataframe at col_to_count - date level
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

@Timer(logger = logging.info)
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
    df[f"{'_'.join(levels)}_qty_sold_day"].fillna(0, inplace=True)
    return df

@Timer(logger = logging.info)
def add_zero_qty_after_last_date_rows(df, test_df, levels):
    """Add new (missing) rows between last observed sale date and last day of
    the training period for each value of levels and only for levels that exist
    in the test data, with 0 values for quantity sold.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe in which to create new rows
    test_df : pandas DataFrame
        Existing shop-item level dataframe for the test period
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
        .reset_index()
        .rename(columns={"date": f"last_{'_'.join(levels)}_date"})
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
    addl_dates = addl_dates.groupby(addl_dates.index).apply(
        lambda x: x.set_index("date").resample("D").asfreq()
    )
    addl_dates.reset_index(inplace=True)
    if len(levels) == 2:
        addl_dates[levels[0] + "_id"], addl_dates[levels[1] + "_id"] = zip(
            *addl_dates.level_0
        )
        addl_dates.drop("level_0", axis=1, inplace=True)
    addl_dates.query("date != datetime.datetime(*FIRST_DAY_OF_TEST_PRD)", inplace=True)
    addl_dates = addl_dates.groupby([level + "_id" for level in levels]).apply(
        lambda group: group.iloc[1:,]
    )
    addl_dates.reset_index(drop=True, inplace=True)

    addl_dates = addl_dates.merge(
        test_levels, on=[level + "_id" for level in levels], how="inner"
    )

    df = pd.concat([df, addl_dates], axis=0, ignore_index=True)
    df.sort_values(
        by=[level + "_id" for level in levels] + ["date"],
        inplace=True,
        ignore_index=True,
    )
    df[f"{'_'.join(levels)}_qty_sold_day"].fillna(0, inplace=True)
    return df

@Timer(logger = logging.info)
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
    try:
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
        return df
    except KeyError as e:
        print("prev_nonzero_qty_sold function was not executed due to:")
        print(e)

@Timer(logger = logging.info)
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
    return df

@Timer(logger = logging.info)
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
    return df

@Timer(logger = logging.info)
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
    return df

@Timer(logger = logging.info)
def num_of_sale_dts_in_prev_x_days(df, levels):
    """Create columns for number of days in previous 7-day and 30-day periods
    with a sale, as well as number of days with a sale since start of train period.

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
    df["day_w_sale"] = np.where(df[f"{'_'.join(levels)}_qty_sold_day"] > 0, 1, 0)

    df[f"{'_'.join(levels)}_cnt_sale_dts_last_7d"] = df.groupby(
        [level + "_id" for level in levels]
    )["day_w_sale"].apply(lambda x: x.rolling(7, min_periods=1).sum().shift().fillna(0))
    df[f"{'_'.join(levels)}_cnt_sale_dts_last_30d"] = df.groupby(
        [level + "_id" for level in levels]
    )["day_w_sale"].apply(
        lambda x: x.rolling(30, min_periods=1).sum().shift().fillna(0)
    )
    df[f"{'_'.join(levels)}_cnt_sale_dts_before_day"] = df.groupby(
        [level + "_id" for level in levels]
    )["day_w_sale"].apply(lambda x: x.expanding().sum().shift().fillna(0))

    df.drop("day_w_sale", axis=1, inplace=True)
    return df

@Timer(logger = logging.info)
def rolling_7d_qty_stats(df, levels):
    """Create rolling max, min, mean, mode and median quantity sold values,
    grouped by values of specified column(s).

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
    df[f"{'_'.join(levels)}_rolling_7d_max_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).max().shift().bfill()
    )
    df[f"{'_'.join(levels)}_rolling_7d_min_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).min().shift().bfill()
    )
    df[f"{'_'.join(levels)}_rolling_7d_avg_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).mean().shift().bfill()
    )

    df[f"{'_'.join(levels)}_rolling_7d_mode_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1)
        .agg(lambda x: x.value_counts().index[0])
        .shift()
        .bfill()
    )

    df[f"{'_'.join(levels)}_rolling_7d_median_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).median().shift().bfill()
    )
    return df

@Timer(logger = logging.info)
def expanding_cv2_of_qty(df, levels):
    """Create column for expanding coefficient of variation squared of quantity
    bought before current day, with only non-zero quantity values considered.

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
    df["expand_cv2_of_qty"] = (
        df.groupby([level + "_id" for level in levels])[
            f"{'_'.join(levels)}_qty_sold_day"
        ]
        .expanding()
        .apply(
            lambda x: np.square(
                bn.nanstd(np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0)))
                / bn.nanmean(np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0)))
            ),
            raw=True,
        )
    )
    return df

@Timer(logger = logging.info)
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
    df[f"{'_'.join(levels)}_expanding_adi"] = (
        df[f"{'_'.join(levels)}_days_since_first_sale"]
        .div(df[f"{'_'.join(levels)}_cnt_sale_dts_before_day"])
        .replace(np.inf, 0)
    )
    return df

@Timer(logger = logging.info)
def expanding_qty_sold_stats(df, levels):
    """Create expanding max, min, mean, mode and median quantity sold values,
    grouped by values of specified column(s).

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
    df[f"{'_'.join(levels)}_expand_qty_max"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.expanding().max().shift().bfill()
    )

    df[f"{'_'.join(levels)}_expand_qty_min"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.expanding().min().shift().bfill()
    )

    df[f"{'_'.join(levels)}_expand_qty_mean"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.expanding().mean().shift().bfill()
    )

    df[f"{'_'.join(levels)}_expand_qty_mode"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.expanding().agg(lambda x: x.value_counts().index[0]).shift().bfill()
    )

    df[f"{'_'.join(levels)}_expand_qty_median"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_qty_sold_day"].apply(
        lambda x: x.expanding().median().shift().bfill()
    )
    return df

@Timer(logger = logging.info)
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
    for shift_val in [1, 2, 3]:
        df[f"{'_'.join(levels)}_qty_sold_{shift_val}d_ago"] = df.groupby(
            [level + "_id" for level in levels]
        )[f"{'_'.join(levels)}_qty_sold_day"].shift(shift_val)
        df[f"{'_'.join(levels)}_qty_sold_{shift_val}d_ago"].fillna(0, inplace=True)
    return df

@Timer(logger = logging.info)
def qty_sold_same_day_prev_week(df, levels):
    """Create column for quantity sold same day previous week.

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
    col_list = (
        [level + "_id" for level in levels]
        + ["date"]
        + [f"{'_'.join(levels)}_qty_sold_day"]
    )
    date_plus7_df = df[col_list]
    date_plus7_df["date_plus7"] = date_plus7_df["date"] + datetime.timedelta(days=7)

    date_plus7_df.drop(columns="date", inplace=True)
    date_plus7_df.rename(
        columns={
            f"{'_'.join(levels)}_qty_sold_day": f"{'_'.join(levels)}_qty_sold_last_dow",
            "date_plus7": "date",
        },
        inplace=True,
    )

    df = df.merge(date_plus7_df, on=col_list[:-1], how="left")
    df[f"{'_'.join(levels)}_qty_sold_last_dow"].fillna(0, inplace=True)
    return df

@Timer(logger = logging.info)
def expanding_time_bw_sales_stats(df, levels):
    """Create expanding max, min, mean, mode, median and standard deviation of
    quantity sold values columns, grouped by values of specified column(s).

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
    df[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"] = np.where(
        df[f"{'_'.join(levels)}_qty_sold_day"] > 0,
        df[f"{'_'.join(levels)}_days_since_prev_sale"],
        np.nan,
    )
    df[f"{'_'.join(levels)}_date_max_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().max().shift().bfill()
    )
    df[f"{'_'.join(levels)}_date_min_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().min().shift().bfill()
    )
    df[f"{'_'.join(levels)}_date_avg_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().mean().shift().bfill()
    )
    df[f"{'_'.join(levels)}_date_median_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().median().shift().bfill()
    )
    df[f"{'_'.join(levels)}_date_mode_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().agg(lambda x: x.value_counts().index[0]).shift().bfill()
    )
    df[f"{'_'.join(levels)}_date_std_gap_bw_sales"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_days_since_prev_sale_lmtd"].apply(
        lambda x: x.expanding().std(ddof=0).shift().bfill()
    )
    df.drop(f"{'_'.join(levels)}_days_since_prev_sale_lmtd", axis=1, inplace=True)
    return df

@Timer(logger = logging.info)
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
    non_zero_qty_level_dates = df.query(f"{'_'.join(levels)}_qty_sold_day != 0")[
        [level + "_id" for level in levels]
        + ["date"]
        + [f"{'_'.join(levels)}_qty_sold_day"]
    ]
    last_date_per_level = (
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

    df = pd.concat(
        [
            df,
            non_zero_qty_level_dates.query(
                "date == datetime.datetime(*FIRST_DAY_OF_TEST_PRD)"
            )[[level + "_id" for level in levels] + ["date"]],
        ],
        axis=0,
        ignore_index=True,
    )
    df = df.merge(
        non_zero_qty_level_dates,
        on=[level + "_id" for level in levels] + ["date"],
        how="left",
    )
    df[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"] = df.groupby(
        [level + "_id" for level in levels]
    )[f"{'_'.join(levels)}_date_diff_bw_last_and_prev_qty"].fillna(method="bfill")
    df.query(
        "date != datetime.datetime(*FIRST_DAY_OF_TEST_PRD)", inplace=True
    ).reset_index(drop=True)
    return df

@Timer(logger = logging.info)
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

    sales_sorted = sales[col_list].sort_values(by=[level, "date"], ignore_index=True)

    group_dates_w_val_cts = sales_sorted[col_list.remove(level)]
    group_dates_w_val_cts.index = [
        sales_sorted[level],
        sales_sorted.groupby(level).cumcount().rename("iidx"),
    ]

    group_dates_w_val_cts = (
        group_dates_w_val_cts.groupby(level=level)
        .apply(
            _lag_merge_asof,
            [col for col in col_list if col not in (level, "date")][0],
            lag=1,
        )
        .reset_index(level)
        .reset_index(drop=True)
        .rename(
            columns={
                "num_unique_values_prior_to_day": f"num_unique_{[col for col in col_list if col not in (level, 'date')][0].replace('_id','s')}_prior_to_day"
            },
            axis=1,
        )
    )
    group_dates_w_val_cts.drop_duplicates(subset=[level, "date"], inplace=True)
    group_dates_w_val_cts[
        f"num_unique_{[col for col in col_list if col not in (level, 'date')][0].replace('_id','s')}_prior_to_day"
    ].fillna(0, inplace=True)
    group_dates_w_val_cts.drop(
        [col for col in col_list if col not in (level, "date")][0], axis=1, inplace=True
    )

    df = df.merge(group_dates_w_val_cts, on=[level, "date"], how="left")
    return df

@Timer(logger = logging.info)
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
    """
    max_qty_by_group_date = df.groupby([level + "_id" for level in levels]).apply(
        lambda x: x.set_index("date")[f"{'_'.join(levels)}_qty_sold_day"]
        .expanding()
        .max()
    )
    df["date_of_max_qty"] = (
        max_qty_by_group_date.groupby(max_qty_by_group_date)
        .transform("idxmax")
        .apply(lambda x: pd.to_datetime(x[len(levels)]))
        .values
    )
    df["date_of_max_qty"] = df.groupby(
        [level + "_id" for level in levels]
    ).date_of_max_qty.shift()
    df.loc[df.date_of_max_qty.isnull(), "date_of_max_qty"] = df.date
    df["days_since_max_qty_sold"] = (df.date - df.date_of_max_qty).dt.days
    df.drop("date_of_max_qty", axis=1, inplace=True)
    return df

@Timer(logger = logging.info)
def build_item_date_lvl_features(test_df, items_df, return_df=False, to_sql=False):
    """Build dataframe of item-date-level features.

    Parameters:
    -----------
    test_df : pandas DataFrame
        Existing dataframe with shop_id-item_id combinations for test period
    items_df : pandas DataFrame
        Existing dataframe with item_id to item_category_id mapping
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of item_id and date and columns containing item-date-level features
    """
    # check if cleaned sales file already exists
    cleaned_sales_file = Path("sales_cleaned.csv")
    if cleaned_sales_file.is_file():
        sales = pd.read_csv("sales_cleaned.csv")
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(sales, return_df=True)

    # Quantity Sold

    # create column with quantity sold by item-date
    item_date_level_features = (
        sales.groupby(["item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "item_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    (
        item_date_level_features.pipe(add_zero_qty_rows, ["item"])
        .pipe(add_zero_qty_after_last_date_rows, test_df, ["item"])
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
        .pipe(qty_sold_same_day_prev_week, ["item"])
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
        (
            item_date_level_features[["date", "item_category_id", "item_qty_sold_day"]]
            .groupby("item_category_id")
            .apply(lambda x: x.resample("D", on="date").item_qty_sold_day.sum())
        )
        .stack()
        .reset_index(name="cat_qty_sold_day")
    )

    # calculate rolling weekly sum of quantity sold for each category, excluding current date
    cat_date_total_qty["cat_qty_sold_last_7d"] = cat_date_total_qty.groupby(
        "item_category_id"
    )["cat_qty_sold_day"].apply(lambda x: x.rolling(7, 1).sum().shift().fillna(0))

    # merge rolling weeekly category quantity totals onto item-date dataset
    item_date_level_features = item_date_level_features.merge(
        cat_date_total_qty[["item_category_id", "date", "cat_qty_sold_last_7d"]],
        on=["item_category_id", "date"],
        how="left",
    )

    # create dataframe with daily lists of items sold for each category
    cat_date_item_lists = (
        item_date_level_features[["date", "item_category_id", "item_id"]]
        .groupby("item_category_id")
        .apply(lambda x: x.set_index("date").item_id.agg(list))
        .reset_index()
        .rename(columns={0: "item_list"})
    )

    # create column with rolling weekly count of unique items sold for each category, excluding current date

    cat_grouped = cat_date_item_lists.groupby("item_category_id")
    cat_date_item_lists["cat_unique_items_sold_last_7d"] = np.nan

    # iterate over each group
    for cat_name, cat_group in cat_grouped:
        cat_group_dt_idx = cat_group.set_index("date")
        # list to hold rolling accumulated lists of items for rolling 1-week period
        rolllists = []
        # iterate over rows within each group
        for row_index, _ in cat_group_dt_idx.iterrows():
            # list to hold accumulated list for one day, including day of
            res = []
            for d in pd.date_range(
                start=max(
                    cat_group_dt_idx.index.min() - datetime.timedelta(1),
                    row_index - datetime.timedelta(7),
                ),
                end=row_index - datetime.timedelta(1),
            ):
                res.append(cat_group_dt_idx.loc[d + datetime.timedelta(1), "item_list"])
            rolllists.append(res)
        for idx, li in enumerate(rolllists):
            li = [item for sublist in li for item in sublist]
            rolllists[idx] = len({*li})
        cat_date_item_lists.loc[
            cat_date_item_lists.item_category_id == cat_name,
            "cat_unique_items_sold_last_7d",
        ] = rolllists

    cat_date_item_lists["cat_unique_items_sold_last_7d"] = (
        cat_date_item_lists.groupby("item_category_id")["cat_unique_items_sold_last_7d"]
        .shift()
        .fillna(0)
    )

    # merge rolling weekly category-grouped counts of unique items onto item-date dataset
    item_date_level_features = item_date_level_features.merge(
        cat_date_item_lists[
            ["item_category_id", "date", "cat_unique_items_sold_last_7d"]
        ],
        on=["item_category_id", "date"],
        how="left",
    )

    # add column with quantity sold per item in category in the last week
    item_date_level_features["cat_qty_sold_per_item_last_7d"] = (
        item_date_level_features["cat_qty_sold_last_7d"]
        .div(item_date_level_features["cat_unique_items_sold_last_7d"])
        .replace(np.inf, 0)
    )

    # Presence of Spikes in Quantity Sold

    # column indicating whether item had a spike in quantity sold before current day
    res = (
        item_date_level_features.groupby("item_id")["item_qty_sold_day"]
        .expanding()
        .apply(
            lambda x: any(
                (
                    v
                    > (
                        bn.nanmedian(
                            np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0))
                        )
                        + 2
                        * bn.nanstd(
                            np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0))
                        )
                    )
                )
                for v in x[:-1]
            ),
            raw=True,
        )
    )
    item_date_level_features["item_had_spike_before_day"] = np.where(res == 1, 1, 0)

    # column with count of spikes in quantity sold before current day

    res = (
        item_date_level_features.groupby("item_id")["item_qty_sold_day"]
        .expanding()
        .apply(
            lambda x: sum(
                (
                    v
                    > (
                        bn.nanmedian(
                            np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0))
                        )
                        + 2
                        * bn.nanstd(
                            np.ma.MaskedArray(x[:-1], mask=(np.array(x[:-1]) == 0))
                        )
                    )
                )
                for v in x[:-1]
            ),
            raw=True,
        )
    ).reset_index(level=0, drop=True)
    item_date_level_features["item_had_spike_before_day"] = res

    item_date_level_features = _downcast(item_date_level_features)
    item_date_level_features = _add_col_prefix(item_date_level_features, "id_")

    item_date_level_features.rename(columns={'date':'sale_date'}, inplace=True)

    if to_sql:
        write_df_to_sql(item_date_level_features, "item_dates")

    if return_df:
        return item_date_level_features


# SHOP-DATE-LEVEL FEATURES
@Timer(logger = logging.info)
def build_shop_date_lvl_features(test_df, items_df, return_df=False, to_sql=False):
    """Build dataframe of shop-date-level features.

    Parameters:
    -----------
    test_df : pandas DataFrame
        Existing dataframe with shop_id-item_id combinations for test period
    items_df : pandas DataFrame
        Existing dataframe with item_id to item_category_id mapping
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    shop_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id and date and columns containing shop-date-level features
    """
    # check if cleaned sales file already exists
    cleaned_sales_file = Path("sales_cleaned.csv")
    if cleaned_sales_file.is_file():
        sales = pd.read_csv("sales_cleaned.csv")
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(sales, return_df=True)

    # Quantity Sold by Shop-Date

    # create column with quantity sold by shop-date
    shop_date_level_features = (
        sales.groupby(["shop_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    (
        shop_date_level_features.pipe(add_zero_qty_rows, ["shop"])
        .pipe(add_zero_qty_after_last_date_rows, test_df, ["shop"])
        .pipe(prev_nonzero_qty_sold, ["shop"])
        .pipe(days_elapsed_since_prev_sale, ["shop"])
        .pipe(days_elapsed_since_first_sale, ["shop"])
        .pipe(first_week_month_of_sale, ["shop"])
        .pipe(num_of_sale_dts_in_prev_x_days, ["shop"])
        .pipe(rolling_7d_qty_stats, ["shop"])
        .pipe(expanding_qty_sold_stats, ["shop"])
        .pipe(qty_sold_x_days_before, ["shop"])
        .pipe(qty_sold_same_day_prev_week, ["shop"])
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
            },
            axis=1,
        )
    )
    shop_dates_w_item_cat_cts.drop_duplicates(subset=["shop_id", "date"], inplace=True)
    shop_dates_w_item_cat_cts.num_unique_item_cats_prior_to_day.fillna(0, inplace=True)
    shop_dates_w_item_cat_cts.drop("item_category_id", axis=1, inplace=True)

    shop_date_level_features = shop_date_level_features.merge(
        shop_dates_w_item_cat_cts, on=["shop_id", "date"], how="left"
    )

    shop_date_level_features = _downcast(shop_date_level_features)
    shop_date_level_features = _add_col_prefix(shop_date_level_features, "sd_")

    shop_date_level_features.rename(columns={'date':'sale_date'}, inplace=True)

    if to_sql:
        write_df_to_sql(shop_date_level_features, "shop_dates")

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


# SHOP-ITEM-DATE-LEVEL FEATURES
@Timer(logger = logging.info)
def build_shop_item_date_lvl_features(test_df, items_df, return_df=False, to_sql=False):
    """Build dataframe of shop-item-date-level features.

    Parameters:
    -----------
    test_df : pandas DataFrame
        Existing dataframe with shop_id-item_id combinations for test period
    items_df : pandas DataFrame
        Existing dataframe with item_id to item_category_id mapping
    return_df : bool
        Return resulting dataframe (True) or not (False)
    to_sql : bool
        Write resulting dataframe to SQL table (True) or not (False)

    Returns:
    --------
    shop_item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id, item_id, and date and columns containing shop-item-date-level features
    """
    # check if cleaned sales file already exists
    cleaned_sales_file = Path("sales_cleaned.csv")
    if cleaned_sales_file.is_file():
        sales = pd.read_csv("sales_cleaned.csv")
        sales["date"] = pd.to_datetime(sales.date)
    else:
        sales = clean_sales_data(sales, return_df=True)

    # Total Quantity Sold by Shop-Item-Date

    # create column with quantity sold by shop-item-date
    shop_item_date_level_features = (
        sales.groupby(["shop_id", "item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_item_qty_sold_day"})
    )

    # pass dataframe through multiple functions to add needed rows and columns
    (
        shop_item_date_level_features.pipe(add_zero_qty_rows, ["shop", "item"])
        .pipe(add_zero_qty_after_last_date_rows, test_df, ["shop", "item"])
        .pipe(prev_nonzero_qty_sold, ["shop", "item"])
        .pipe(days_elapsed_since_prev_sale, ["shop", "item"])
        .pipe(days_elapsed_since_first_sale, ["shop", "item"])
        .pipe(first_week_month_of_sale, ["shop", "item"])
        .pipe(num_of_sale_dts_in_prev_x_days, ["shop", "item"])
        .pipe(rolling_7d_qty_stats, ["shop", "item"])
        .pipe(expanding_cv2_of_qty, ["shop", "item"])
        .pipe(expanding_avg_demand_int, ["shop", "item"])
        .pipe(expanding_qty_sold_stats, ["shop", "item"])
        .pipe(qty_sold_x_days_before, ["shop", "item"])
        .pipe(qty_sold_same_day_prev_week, ["shop", "item"])
        .pipe(expanding_time_bw_sales_stats, ["shop", "item"])
        .pipe(diff_bw_last_and_sec_to_last_qty, ["shop", "item"])
        .pipe(days_since_max_qty_sold, ["shop", "item"])
    )

    # Expanding Coefficient of Variation of item price (across all dates for shop-item before current date)

    shop_item_date_level_features["coef_var_price"] = sales.groupby(
        ["shop_id", "item_id"]
    )["item_price"].apply(lambda x: x.expanding().agg(variation).shift().bfill())

    # Expanding Mean Absolute Deviation of Quantity Sold (across all shop-items before current date)

    shop_item_date_level_features["quant_mean_abs_dev"] = sales.groupby(
        ["shop_id", "item_id"]
    )["item_cnt_day"].apply(lambda x: x.expanding().agg(_mad).shift().bfill())

    # Expanding Median Absolute Deviation of Quantity Sold (across all shop-items before current date)

    shop_item_date_level_features["quant_median_abs_dev"] = sales.groupby(
        ["shop_id", "item_id"]
    )["item_cnt_day"].apply(
        lambda x: x.expanding().agg(median_absolute_deviation).shift().bfill()
    )

    # Demand for Category in Last Week (Quantity Sold)
    # also, flag for whether any items in same category were sold at the shop before current day

    # Add item_category_id column
    shop_item_date_level_features = shop_item_date_level_features.merge(
        items_df[["item_id", "item_category_id"]], on="item_id", how="left"
    )

    # create dataframe with daily totals of quantity sold for each category at each shop
    shop_cat_date_total_qty = (
        shop_item_date_level_features[
            ["shop_id", "date", "item_category_id", "shop_item_qty_sold_day"]
        ]
        .groupby(["shop_id", "item_category_id"])
        .apply(lambda x: x.resample("D", on="date").shop_item_qty_sold_day.sum())
    ).reset_index(name="shop_cat_qty_sold_day")

    # calculate rolling weekly sum of quantity sold for each category at each shop, excluding current date
    shop_cat_date_total_qty[
        "shop_cat_qty_sold_last_7d"
    ] = shop_cat_date_total_qty.groupby(["shop_id", "item_category_id"])[
        "shop_cat_qty_sold_day"
    ].apply(
        lambda x: x.rolling(7, 1).sum().shift().fillna(0)
    )

    shop_cat_date_total_qty["cat_sold_at_shop_before_day_flag"] = (
        shop_cat_date_total_qty.groupby(["shop_id", "item_category_id"])[
            "shop_cat_qty_sold_day"
        ]
        .apply(lambda x: x.expanding().sum().shift().fillna(0))
        .astype(bool)
        .astype(np.int8)
    )

    # merge rolling weekly category quantity totals and flag column onto shop-item-date dataset
    shop_item_date_level_features = shop_item_date_level_features.merge(
        shop_cat_date_total_qty[
            [
                "shop_id",
                "item_category_id",
                "date",
                "shop_cat_qty_sold_last_7d",
                "cat_sold_at_shop_before_day_flag",
            ]
        ],
        on=["shop_id", "item_category_id", "date"],
        how="left",
    )

    shop_item_date_level_features = _downcast(shop_item_date_level_features)
    shop_item_date_level_features = _add_col_prefix(
        shop_item_date_level_features, "sid_"
    )

    shop_item_date_level_features.rename(columns={'date':'sale_date'}, inplace=True)

    if to_sql:
        write_df_to_sql(shop_item_date_level_features, "shop_item_dates")

    if return_df:
        return shop_item_date_level_features


def main():
    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir_file = "./logging.log"

    logging.basicConfig(level=logging.DEBUG,
        filemode='w',
        format=fmt,
        datefmt=datefmt,
        filename=log_dir_file)

    logger = logging.getLogger()

    logger.info(f"The Python version is {platform.python_version()}.")
    logger.info(f"The pandas version is {pd.__version__}.")
    logger.info(f"The numpy version is {np.__version__}.")
    logger.info(f"The scipy version is {scipy.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", metavar="<command>", help="'create', 'start' or 'stop'",
    )
    parser.add_argument(
        "--send_to_sql",
        default=False,
        action="store_true",
        help="write DF to SQL (if included) or not (if not included)",
    )

    args = parser.parse_args()

    # Load data
    data_path = "./Data/competitive-data-science-predict-future-sales/"

    sales = pd.read_csv(data_path + "sales_train.csv")
    test_df = pd.read_csv(data_path + "test.csv")
    items_df = pd.read_csv(data_path + "items.csv")
    categories_df = pd.read_csv(data_path + "item_categories.csv")
    shops_df = pd.read_csv(data_path + "shops.csv")
    macro_df = pd.read_csv(data_path + "macro.csv")

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

    # Call clean_sales_data() if need to run that code in standalone fashion.
    # If calling other functions that rely on clean sales data, check if clean
    # data file exists and use it if it does or call clean_sales_data() if
    # it does not.

    if args.command == "clean":
        clean_sales_data(sales, to_sql=args.send_to_sql)
    elif args.command == "shops":
        build_shop_lvl_features(shops_df, to_sql=args.send_to_sql)
    elif args.command == "items":
        build_item_lvl_features(items_df, categories_df, to_sql=args.send_to_sql)
    elif args.command == "dates":
        build_date_lvl_features(macro_df, ps4games, sales, to_sql=args.send_to_sql)
    elif args.command == "item-dates":
        build_item_date_lvl_features(test_df, items_df, to_sql=args.send_to_sql)
    elif args.command == "shop-dates":
        build_shop_date_lvl_features(test_df, items_df, to_sql=args.send_to_sql)
    elif args.command == "shop-item-dates":
        build_shop_item_date_lvl_features(test_df, items_df, to_sql=args.send_to_sql)
    else:
        print(
            "'{}' is not recognized. "
            "Use 'clean', 'shops', 'items', 'dates', 'item-dates' "
            "'shop-dates' or 'shop-item-dates'".format(args.command)
        )


if __name__ == "__main__":
    main()
