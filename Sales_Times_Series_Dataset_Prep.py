"""
Sales Time Series Dataset Processing and Database Load
Create features at different levels of analysis and upload to PostgreSQL on AWS RDS.

Copyright (c) 2020 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)
Written by Sasha Kapralov

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tower.py train --dataset=/path/to/tower/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tower.py train --dataset=/path/to/tower/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tower.py train --dataset=/path/to/tower/dataset --weights=imagenet

    # Apply color splash to an image
    python3 tower.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 tower.py splash --weights=last --video=<URL or path to file>
"""

import sys
import datetime
from functools import partial
import pickle
import re
import sqlite3
import time

from constants import (
    FIRST_DAY_OF_TRAIN_PRD,
    LAST_DAY_OF_TRAIN_PRD,
    FIRST_DAY_OF_TEST_PRD,
)

import holidays

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf
from scipy.stats import variation, mode

# set max columns displayed to 100
pd.set_option("display.max_columns", 100)

import warnings

warnings.filterwarnings("ignore")

# PERFORM INITIAL DATA CLEANING
def clean_sales_data():
    """Perform initial data cleaning of the train shop-item-date data.

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
    sales = sales[sales.item_price > 0.0]

    return sales


# SHOP-LEVEL FEATURES


def lat_lon_to_float(
    in_coord, degree_sign=u"\N{DEGREE SIGN}", remove=degree_sign + "′" + "″"
):
    """Convert latitude-longitude text string into latitude and longitude floats.

    Parameters:
    -----------
    in_coord : str
        latitude-longitude string (example format: '55°53′21″ с. ш. 37°26′42″ в. д.')
    degree_sign : str
        Unicode representation of degree sign
    remove : str
        Concatenation of characters to remove from latitude-longitude string

    Returns:
    --------
    geo_lat, geo_lon: float values of latitude and longitude (i.e., with minutes and seconds converted to decimals)
    """
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


def build_shop_lvl_features():
    """Build dataframe of shop-level features.

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

    # city populations as of 1/1/2020
    # (source: https://rosstat.gov.ru/storage/mediabank/CcG8qBhP/mun_obr2020.rar, accessed 11/17/2020):
    city_pop = [
        ("РостовНаДону", 1137904.0, "47°14′26″ с. ш. 39°42′38″ в. д.", "UTC+3"),
        ("Н.Новгород", 1252236.0, "56°19′37″ с. ш. 44°00′27″ в. д.", "UTC+3"),
        ("Казань", 1257391.0, "55°47′27″ с. ш. 49°06′52″ в. д.", "UTC+3"),
        ("Новосибирск", 1625631.0, "55°01′ с. ш. 82°55′ в. д.", "UTC+7"),
        ("Воронеж", 1058261.0, "51°40′18″ с. ш. 39°12′38″ в. д.", "UTC+3"),
        ("Красноярск", 1093771.0, "56°00′43″ с. ш. 92°52′17″ в. д.", "UTC+7"),
        ("Ярославль", 608353.0, "57°37′ с. ш. 39°51′ в. д.", "UTC+3"),
        ("Тюмень", 807271.0, "57°09′ с. ш. 65°32′ в. д.", "UTC+5"),
        ("Сургут", 380632.0, "61°15′00″ с. ш. 73°26′00″ в. д.", "UTC+5"),
        ("Омск", 1154507.0, "54°58′ с. ш. 73°23′ в. д.", "UTC+6"),
        ("Волжский", 323906.0, "48°47′ с. ш. 44°46′ в. д.", "UTC+4"),
        ("Уфа", 1128787.0, "54°44′ с. ш. 55°58′ в. д.", "UTC+5"),
        ("Якутск", 322987.0, "62°01′38″ с. ш. 129°43′55″ в. д.", "UTC+9"),
        ("Балашиха", 507366.0, "55°48′ с. ш. 37°56′ в. д.", "UTC+3"),
        ("Вологда", 310302.0, "59°13′ с. ш. 39°54′ в. д.", "UTC+3"),
        ("Жуковский", 107560.0, "55°36′04″ с. ш. 38°06′58″ в. д.", "UTC+3"),
        ("Калуга", 332039.0, "54°32′00″ с. ш. 36°16′00″ в. д.", "UTC+3"),
        ("Коломна", 140129.0, "55°05′38″ с. ш. 38°46′05″ в. д.", "UTC+3"),
        ("Курск", 452976.0, "51°43′ с. ш. 36°11′ в. д.", "UTC+3"),
        ("Москва", 12678079.0, "55°45′21″ с. ш. 37°37′04″ в. д.", "UTC+3"),
        ("Мытищи", 235504.0, "55°55′ с. ш. 37°44′ в. д.", "UTC+3"),
        ("СПб", 5398064.0, "59°57′ с. ш. 30°19′ в. д.", "UTC+3"),
        ("Самара", 1156659.0, "53°11′ с. ш. 50°07′ в. д.", "UTC+4"),
        ("Сергиев", 100335.0, "56°18′00″ с. ш. 38°08′00″ в. д.", "UTC+3"),
        ("Томск", 576624.0, "56°29′19″ с. ш. 84°57′08″ в. д.", "UTC+7"),
        ("Химки", 259550.0, "55°53′21″ с. ш. 37°26′42″ в. д.", "UTC+3"),
        ("Чехов", 73321.0, "55°08′42″ с. ш. 37°27′20″ в. д.", "UTC+3"),
        ("Адыгея", 932629.0, "45°02′00″ с. ш. 38°59′00″ в. д.", "UTC+3"),
    ]

    city_df = pd.DataFrame(
        city_pop, columns=["city", "population", "geo_coords", "time_zone"]
    )

    all_lat_lons = city_df.geo_coords.apply(lat_lon_to_float)

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

    # Total Transaction Counts (at Item-Date Level) by Shop

    # Add column for total transaction counts (at item-date level) by shop
    shop_total_tx_cts = (
        sales.groupby("shop_id")
        .size()
        .reset_index()
        .rename(columns={0: "shop_total_tx_cnt"})
    )
    shops_df = shops_df.merge(shop_total_tx_cts, on="shop_id", how="left")

    # Number of Unique Items Sold at Shop

    # Add column for number of unique items sold by shop
    unique_items_by_shop = sales.groupby("shop_id")["item_id"].unique()
    unique_item_cts_by_shop = (
        unique_items_by_shop.map(len)
        .reset_index()
        .rename(columns={"item_id": "shop_unique_item_cts"})
    )
    shops_df = shops_df.merge(unique_item_cts_by_shop, on="shop_id", how="left")

    # Number of Unique Categories of Items Sold at Shop

    # Add column for number of unique categories of items sold by shop
    sales_w_item_cat_id = sales[["item_id", "shop_id",]].merge(
        items_df, on="item_id", how="left"
    )
    unique_cat_cts_by_shop = (
        sales_w_item_cat_id.groupby("shop_id")["item_category_id"]
        .nunique()
        .reset_index()
        .rename(columns={"item_category_id": "shop_unique_cat_cts"})
    )
    shops_df = shops_df.merge(unique_cat_cts_by_shop, on="shop_id", how="left")

    return shops_df


# ITEM-LEVEL FEATURES


def group_game_consoles(cat_name):
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


std_funct = partial(np.std, ddof=0)


def build_item_lvl_features():
    """Build dataframe of item-level features.

    Returns:
    --------
    item_level_features : pandas dataframe
        Dataframe with each row representing an item and columns containing item-level features
    """

    # Coefficient of Variation of Price

    # Calculate the coefficient of variation of price for each item separately
    item_level_features = (
        sales.groupby("item_id")["item_price"]
        .agg(variation)
        .reset_index()
        .rename(columns={"item_price": "coef_var_price"})
    )

    # Mean Absolute Deviation of Quantity Sold

    # Calculate the mean absolute deviation of quantity sold for each item
    item_level_features["quant_mean_abs_dev"] = (
        sales.groupby("item_id")["item_cnt_day"].mad().values
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
    ] = item_level_features.item_category_name.apply(group_game_consoles)

    # Indicator for Digital Item

    # Create indicator column for whether item is digital
    item_level_features["digital_item"] = np.where(
        (item_level_features.item_category_name.str.contains("Цифра"))
        | (item_level_features.item_category_name.str.contains("MP3")),
        1,
        0,
    )

    # Number of Shops That Sold the Item

    # create column for number of shops that sold the item
    item_n_shops_selling = (
        sales.groupby("item_id")
        .shop_id.nunique()
        .reset_index()
        .rename(columns={"shop_id": "item_unique_shop_cts"})
    )
    item_level_features = item_level_features.merge(
        item_n_shops_selling, on="item_id", how="left"
    )

    # Number of Unique Days on Which Item Was Sold

    # create column for number of unique days on which item was sold
    item_n_days_sold = (
        sales.groupby("item_id")["date"]
        .nunique()
        .reset_index()
        .rename(columns={"date": "item_unique_date_cts"})
    )
    item_level_features = item_level_features.merge(
        item_n_days_sold, on="item_id", how="left"
    )

    # Values Summarizing Number of Days Between Sale of Item

    item_dates = (
        sales[["item_id", "date"]]
        .drop_duplicates(subset=["item_id", "date"])
        .sort_values(by=["item_id", "date"])
        .reset_index(drop=True)
    )

    # create column for time elapsed since previous sale of same item
    item_dates["days_since_prev_sale"] = (
        item_dates.groupby("item_id").date.diff().dt.days
    )

    item_gap_bw_days_stats = (
        item_dates.groupby("item_id")
        .agg(
            item_gap_days_mean=("days_since_prev_sale", np.mean),
            item_gap_days_std=("days_since_prev_sale", std_funct),
            item_gap_days_median=("days_since_prev_sale", np.median),
            item_gap_days_min=("days_since_prev_sale", np.min),
            item_gap_days_max=("days_since_prev_sale", np.max),
        )
        .reset_index()
    )

    item_level_features = item_level_features.merge(
        item_gap_bw_days_stats, on="item_id", how="left"
    )

    # Total Number of Items in Category That Includes the Item

    # add column for number of items in each category, with the number assigned to each item in that category
    item_level_features["item_total_items_in_cat"] = item_level_features.groupby(
        "item_category_id"
    ).item_id.transform("count")

    # Presence of Spikes in Quantity Sold

    # add column for item having a spike in quantity sold at day level
    items_w_spike = sales[
        sales.item_cnt_day
        > (
            sales.groupby("item_id").item_cnt_day.transform("mean")
            + 2 * sales.groupby("item_id").item_cnt_day.transform("std", ddof=0)
        )
    ].item_id.unique()
    item_level_features["item_had_qty_spike"] = np.where(
        item_level_features.item_id.isin(items_w_spike), 1, 0
    )

    # add column for number of spikes in quantity sold at day level
    n_spikes_by_item = (
        sales[
            sales.item_cnt_day
            > (
                sales.groupby("item_id").item_cnt_day.transform("mean")
                + 2 * sales.groupby("item_id").item_cnt_day.transform("std", ddof=0)
            )
        ]
        .groupby("item_id")
        .size()
        .reset_index()
        .rename(columns={0: "item_n_spikes"})
    )
    item_level_features = item_level_features.merge(
        n_spikes_by_item, on="item_id", how="left"
    )
    item_level_features.item_n_spikes.fillna(0, inplace=True)

    # add indicator column for item having a spike when the item just went on sale
    qty_spiked = sales.item_cnt_day > (
        sales.groupby("item_id").item_cnt_day.transform("mean")
        + 2 * sales.groupby("item_id").item_cnt_day.transform("std", ddof=0)
    )
    first_week = (sales.date - sales.groupby("item_id").date.min()).astype(
        "timedelta64[D]"
    ) <= 7.0

    items_w_early_spike = sales[qty_spiked & first_week].item_id.unique()
    item_level_features["item_had_early_spike"] = np.where(
        item_level_features.item_id.isin(items_w_early_spike), 1, 0
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

    return item_level_features


# DATE-LEVEL FEATURES

# Russian public holidays in 2012, 2013, 2014, 2015, and 2016
public_holidays = holidays.Russia(years=[2012, 2013, 2014, 2015, 2016])
public_holiday_dts = list(public_holidays.keys())


def days_to_holiday(curr_dt, list_of_holidays=public_holiday_dts):
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


def days_after_holiday(curr_dt, list_of_holidays=public_holiday_dts):
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


def month_counter(d1):
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
        (d1.year - FIRST_DAY_OF_TRAIN_PRD.year) * 12
        + d1.month
        - FIRST_DAY_OF_TRAIN_PRD.month
    )


def build_date_lvl_features():
    """Build dataframe of date-level features.

    Returns:
    --------
    date_level_features : pandas dataframe
        Dataframe with each row representing a date and columns containing date-level features
    """

    # Dates from Start to End of Training Period
    date_level_features = pd.DataFrame(
        {"date": pd.date_range("01-01-2013", "10-31-2015")}
    )

    # create same date_block_num column that exists in sales train dataset
    date_level_features["date_block_num"] = date_level_features.date.apply(
        month_counter
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
        public_holidays
    ).astype(np.int8)

    # Number of Days Before a Holiday and Number of Days After a Holiday

    date_level_features["days_to_holiday"] = date_level_features["date"].apply(
        days_to_holiday
    )
    date_level_features["days_after_holiday"] = date_level_features["date"].apply(
        days_after_holiday
    )

    # Indicator for Major Event

    # create indicator column for major events
    olympics = pd.date_range(start="2/7/2014", end="2/23/2014").to_series().values
    world_cup = pd.date_range(start="6/12/2014", end="7/13/2014").to_series().values
    major_events = np.concatenate([olympics, world_cup])
    date_level_features["major_event"] = date_level_features.date.isin(
        major_events
    ).astype(np.int8)

    # Macroeconomic Indicator Columns

    # convert the date column in macro_df from string to datetime type
    macro_df["date"] = pd.to_datetime(macro_df.timestamp)

    # subset macro_df dataset to relevant period
    macro_df_2013_2015 = macro_df[
        (macro_df.date >= FIRST_DAY_OF_TRAIN_PRD)
        & (macro_df.date <= LAST_DAY_OF_TRAIN_PRD)
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
        ps4games["Release date EU"] <= LAST_DAY_OF_TRAIN_PRD
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
        date_level_features[date_level_features.ps4_game_release_dt == 1]["date"]
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
    date_level_features[
        "day_total_qty_sold_1day_lag"
    ] = date_level_features.day_total_qty_sold.shift(1)
    date_level_features[
        "day_total_qty_sold_6day_lag"
    ] = date_level_features.day_total_qty_sold.shift(6)
    date_level_features[
        "day_total_qty_sold_7day_lag"
    ] = date_level_features.day_total_qty_sold.shift(7)

    # create column for 1-day lagged brent price (based on the results of cross-correlation analysis above)
    date_level_features["brent_1day_lag"] = date_level_features.brent.shift(1)

    return date_level_features


# ITEM-DATE-LEVEL FEATURES


def drange(date_ser):
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


def build_item_date_lvl_features():
    """Build dataframe of item-date-level features.

    Returns:
    --------
    item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of item_id and date and columns containing item-date-level features
    """

    # Quantity Sold

    # create column with quantity sold by item-date
    item_date_level_features = (
        sales.groupby(["item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "item_qty_sold_day"})
    )

    # Add missing item-dates (between first and last dates for each item) with 0 values for quantity sold
    all_item_dates = item_date_level_features.groupby("item_id").date.apply(drange)
    all_item_dates = all_item_dates.reset_index(level=0).reset_index(drop=True)
    item_date_level_features = all_item_dates.merge(
        item_date_level_features, on=["item_id", "date"], how="left"
    )
    item_date_level_features.item_qty_sold_day.fillna(0, inplace=True)

    # Add missing item-dates (between last observed sale date and last day of the training period) for items that exist in test dataset
    test_items = (
        test_df[["item_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by="item_id")
    )
    last_item_dts_in_train_data = (
        item_date_level_features.groupby("item_id")
        .date.max()
        .reset_index()
        .rename(columns={"date": "last_item_date"})
    )

    last_item_dts_in_train_data["last_train_dt"] = FIRST_DAY_OF_TEST_PRD

    addl_dates = (
        last_item_dts_in_train_data.set_index("item_id")
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame()
        .rename(columns={0: "date"})
    )
    addl_dates = addl_dates.groupby(addl_dates.index).apply(
        lambda x: x.set_index("date").resample("D").asfreq()
    )
    addl_dates.reset_index(inplace=True)
    addl_dates = addl_dates[addl_dates.date != FIRST_DAY_OF_TEST_PRD]
    addl_dates = addl_dates.groupby("item_id").apply(lambda group: group.iloc[1:,])
    addl_dates.reset_index(drop=True, inplace=True)

    addl_dates = addl_dates.merge(test_items, on="item_id", how="inner")

    item_date_level_features = pd.concat(
        [item_date_level_features, addl_dates], axis=0, ignore_index=True
    )
    item_date_level_features.sort_values(by=["item_id", "date"], inplace=True)
    item_date_level_features.item_qty_sold_day.fillna(0, inplace=True)

    # Previous Non-Zero Quantity Sold

    item_date_level_features[
        "item_last_qty_sold"
    ] = item_date_level_features.item_qty_sold_day.replace(
        to_replace=0, method="ffill"
    ).shift()
    item_date_level_features.loc[
        item_date_level_features.groupby("item_id")["item_last_qty_sold"].head(1).index,
        "item_last_qty_sold",
    ] = np.NaN
    # fill null values (first item-date) with 0's
    item_date_level_features.item_last_qty_sold.fillna(0, inplace=True)

    # Days Elapsed Since Last Sale of Same Item

    # create column for time elapsed since previous sale of same item
    item_date_level_features["item_last_date"] = np.where(
        item_date_level_features.item_qty_sold_day > 0,
        item_date_level_features.date,
        None,
    )
    item_date_level_features["item_last_date"] = pd.to_datetime(
        item_date_level_features.item_last_date
    )
    item_date_level_features[
        "item_last_date"
    ] = item_date_level_features.item_last_date.fillna(method="ffill").shift()
    item_date_level_features.loc[
        item_date_level_features.groupby("item_id")["item_last_date"].head(1).index,
        "item_last_date",
    ] = pd.NaT
    item_date_level_features[
        "item_days_since_prev_sale"
    ] = item_date_level_features.date.sub(
        item_date_level_features.item_last_date
    ).dt.days
    item_date_level_features.item_days_since_prev_sale.fillna(0, inplace=True)
    item_date_level_features.drop("item_last_date", axis=1, inplace=True)

    # Days Elapsed Since First Date of Sale of Same Item

    # create column for time elapsed since first sale date of same item
    item_date_level_features["item_days_since_first_sale"] = (
        item_date_level_features["date"]
        - item_date_level_features.groupby("item_id")["date"].transform("first")
    ).dt.days

    # Indicator Columns for First Week and First Month of Sale of Item

    # create indicator column for first week of sale of item
    item_date_level_features["item_first_week"] = (
        item_date_level_features["item_days_since_first_sale"] <= 6
    ).astype(np.int8)
    # create indicator column for first month of sale of item
    item_date_level_features["item_first_month"] = (
        item_date_level_features["item_days_since_first_sale"] <= 30
    ).astype(np.int8)

    # Monthly Quantity Summary Values by Item

    item_date_level_features["yr_mon"] = (
        item_date_level_features["date"]
        + pd.offsets.MonthEnd(n=0)
        - pd.offsets.MonthBegin(n=1)
    )
    item_month_qty_stats = (
        item_date_level_features.groupby(["item_id", "yr_mon"])
        .agg(
            item_month_qty_mean=("item_qty_sold_day", np.mean),
            item_month_qty_std=("item_qty_sold_day", std_funct),
            item_month_qty_median=("item_qty_sold_day", np.median),
            item_month_qty_min=("item_qty_sold_day", np.min),
            item_month_qty_max=("item_qty_sold_day", np.max),
        )
        .reset_index()
    )

    item_date_level_features = item_date_level_features.merge(
        item_month_qty_stats, on=["item_id", "yr_mon"], how="left"
    )

    item_date_level_features.drop(columns=["yr_mon"], inplace=True)

    # Rolling 7-day min and max quantity values, excluding current day except for first item-date

    item_date_level_features[
        "item_rolling_7d_max_qty"
    ] = item_date_level_features.groupby("item_id")["item_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).max().shift().bfill()
    )
    item_date_level_features[
        "item_rolling_7d_min_qty"
    ] = item_date_level_features.groupby("item_id")["item_qty_sold_day"].apply(
        lambda x: x.rolling(7, 1).min().shift().bfill()
    )

    # Expanding Max, Min, Mean Quantity Values

    item_date_level_features["item_expand_qty_max"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].apply(lambda x: x.expanding().max().shift().bfill())

    item_date_level_features["item_expand_qty_min"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].apply(lambda x: x.expanding().min().shift().bfill())

    item_date_level_features["item_expand_qty_mean"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].apply(lambda x: x.expanding().mean().shift().bfill())

    # Quantity Sold 1, 2, 3 Days Ago

    item_date_level_features["item_qty_sold_1d_ago"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].shift()
    item_date_level_features.item_qty_sold_1d_ago.fillna(0, inplace=True)

    item_date_level_features["item_qty_sold_2d_ago"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].shift(2)
    item_date_level_features.item_qty_sold_2d_ago.fillna(0, inplace=True)

    item_date_level_features["item_qty_sold_3d_ago"] = item_date_level_features.groupby(
        "item_id"
    )["item_qty_sold_day"].shift(3)
    item_date_level_features.item_qty_sold_3d_ago.fillna(0, inplace=True)

    # Longest Time Interval Between Sales of Items Up to (and Not Including) Current Date

    item_date_level_features["item_days_since_prev_sale_lmtd"] = np.where(
        item_date_level_features.item_qty_sold_day > 0,
        item_date_level_features.item_days_since_prev_sale,
        np.nan,
    )
    item_date_level_features[
        "item_date_max_gap_bw_sales"
    ] = item_date_level_features.groupby("item_id")[
        "item_days_since_prev_sale_lmtd"
    ].apply(
        lambda x: x.expanding().max().shift().bfill()
    )
    item_date_level_features.drop(
        "item_days_since_prev_sale_lmtd", axis=1, inplace=True
    )

    # Difference Between Last and Second-to-Last Quantities Sold

    non_zero_qty_item_dates = item_date_level_features[
        item_date_level_features.item_qty_sold_day != 0
    ][["item_id", "date", "item_qty_sold_day"]]
    last_date_per_item = (
        item_date_level_features[["item_id", "date", "item_qty_sold_day"]]
        .groupby("item_id")
        .tail(1)
        .reset_index(drop=True)
    )
    last_date_per_item = last_date_per_item[last_date_per_item.item_qty_sold_day == 0]
    last_date_per_item["date"] = FIRST_DAY_OF_TEST_PRD
    last_date_per_item["item_qty_sold_day"] = 10
    non_zero_qty_item_dates = pd.concat(
        [non_zero_qty_item_dates, last_date_per_item], axis=0, ignore_index=True
    )
    non_zero_qty_item_dates.sort_values(by=["item_id", "date"], inplace=True)
    non_zero_qty_item_dates["item_date_diff_bw_last_and_prev_qty"] = (
        non_zero_qty_item_dates.groupby("item_id")
        .item_qty_sold_day.diff(periods=2)
        .values
        - non_zero_qty_item_dates.groupby("item_id").item_qty_sold_day.diff().values
    )
    non_zero_qty_item_dates.drop("item_qty_sold_day", axis=1, inplace=True)
    non_zero_qty_item_dates.item_date_diff_bw_last_and_prev_qty.fillna(0, inplace=True)

    item_date_level_features = pd.concat(
        [
            item_date_level_features,
            non_zero_qty_item_dates[
                non_zero_qty_item_dates.date == FIRST_DAY_OF_TEST_PRD
            ][["item_id", "date"]],
        ],
        axis=0,
        ignore_index=True,
    )
    item_date_level_features = item_date_level_features.merge(
        non_zero_qty_item_dates, on=["item_id", "date"], how="left"
    )
    item_date_level_features[
        "item_date_diff_bw_last_and_prev_qty"
    ] = item_date_level_features.groupby(
        "item_id"
    ).item_date_diff_bw_last_and_prev_qty.fillna(
        method="bfill"
    )
    item_date_level_features = item_date_level_features[
        item_date_level_features.date != FIRST_DAY_OF_TEST_PRD
    ].reset_index(drop=True)

    # Demand for Category in Last Week (Quantity Sold, Count of Unique Items Sold, Quantity Sold per Item)

    # Add item_category_id column
    item_date_level_features = item_date_level_features.merge(
        items_df[["item_id", "item_category_id"]], on="item_id", how="left"
    )

    # create dataframe with daily totals of quantity sold for each category
    cat_date_total_qty = (
        item_date_level_features[["date", "item_category_id", "item_qty_sold_day"]]
        .groupby("item_category_id")
        .apply(lambda x: x.set_index("date").item_qty_sold_day.sum())
        .reset_index()
    )

    # calculate rolling weekly sum of quantity sold for each category, excluding current date
    cat_date_total_qty["cat_qty_sold_last_7d"] = cat_date_total_qty.groupby(
        "item_category_id"
    )["item_qty_sold_day"].apply(lambda x: x.rolling(7, 1).sum().shift().fillna(0))

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
            rolllists[idx] = len(set(li))
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

    return item_date_level_features


# SHOP-DATE-LEVEL FEATURES


def build_shop_date_lvl_features():
    """Build dataframe of shop-date-level features.

    Returns:
    --------
    shop_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id and date and columns containing shop-date-level features
    """

    # Quantity Sold by Shop-Date

    # create column with quantity sold by shop-date
    shop_date_level_features = (
        sales.groupby(["shop_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_qty_sold_day"})
    )

    # Add missing shop-dates (between first and last dates for each shop) with 0 values for quantity sold
    all_shop_dates = shop_date_level_features.groupby("shop_id").date.apply(drange)
    all_shop_dates = all_shop_dates.reset_index(level=0).reset_index(drop=True)
    shop_date_level_features = all_shop_dates.merge(
        shop_date_level_features, on=["shop_id", "date"], how="left"
    )
    shop_date_level_features.shop_qty_sold_day.fillna(0, inplace=True)

    # Add missing shop-dates (between last observed sale date and last day of the training period) for shops that exist in test dataset
    test_shops = (
        test_df[["shop_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by="shop_id")
    )
    last_shop_dts_in_train_data = (
        shop_date_level_features.groupby("shop_id")
        .date.max()
        .reset_index()
        .rename(columns={"date": "last_shop_date"})
    )

    last_shop_dts_in_train_data["last_train_dt"] = FIRST_DAY_OF_TEST_PRD

    addl_dates = (
        last_shop_dts_in_train_data.set_index("shop_id")
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame()
        .rename(columns={0: "date"})
    )
    addl_dates = addl_dates.groupby(addl_dates.index).apply(
        lambda x: x.set_index("date").resample("D").asfreq()
    )
    addl_dates.reset_index(inplace=True)
    addl_dates = addl_dates[addl_dates.date != FIRST_DAY_OF_TEST_PRD]
    addl_dates = addl_dates.groupby("shop_id").apply(lambda group: group.iloc[1:,])
    addl_dates.reset_index(drop=True, inplace=True)

    addl_dates = addl_dates.merge(test_shops, on="shop_id", how="inner")

    shop_date_level_features = pd.concat(
        [shop_date_level_features, addl_dates], axis=0, ignore_index=True
    )
    shop_date_level_features.sort_values(by=["shop_id", "date"], inplace=True)
    shop_date_level_features.shop_qty_sold_day.fillna(0, inplace=True)

    # Last Quantity Sold by Shop-Date

    # create lag column for quantity sold, grouped by shop
    shifted = shop_date_level_features.groupby("shop_id").shop_qty_sold_day.shift()
    shop_date_level_features = shop_date_level_features.join(
        shifted.rename("shop_qty_sold_day_lag")
    )
    shop_date_level_features.shop_qty_sold_day_lag.fillna(0, inplace=True)

    # Days Elapsed Since First Sale Date in Same Shop (Age of Shop)

    # create column for time elapsed since first sale date in same shop (i.e., age of shop)
    shop_date_level_features["shop_days_since_first_sale"] = (
        shop_date_level_features["date"]
        - shop_date_level_features.groupby("shop_id")["date"].transform("first")
    ).dt.days

    # Indicator Columns for First Week and First Month of Sale at Shop

    # create indicator column for first week of sale at shop
    shop_date_level_features["shop_first_week"] = (
        shop_date_level_features["shop_days_since_first_sale"] <= 6
    ).astype(np.int8)
    # create indicator column for first month of sale at shop
    shop_date_level_features["shop_first_month"] = (
        shop_date_level_features["shop_days_since_first_sale"] <= 30
    ).astype(np.int8)

    # Monthly Quantity Summary Values by Shop

    shop_date_level_features["yr_mon"] = (
        shop_date_level_features["date"]
        + pd.offsets.MonthEnd(n=0)
        - pd.offsets.MonthBegin(1)
    )
    shop_month_qty_stats = (
        shop_date_level_features.groupby(["shop_id", "yr_mon"])
        .agg(
            shop_month_qty_mean=("shop_qty_sold_day", np.mean),
            shop_month_qty_std=("shop_qty_sold_day", std_funct),
            shop_month_qty_median=("shop_qty_sold_day", np.median),
            shop_month_qty_min=("shop_qty_sold_day", np.min),
            shop_month_qty_max=("shop_qty_sold_day", np.max),
        )
        .reset_index()
    )

    shop_date_level_features = shop_date_level_features.merge(
        shop_month_qty_stats, on=["shop_id", "yr_mon"], how="left"
    )

    shop_date_level_features.drop(columns=["yr_mon"], inplace=True)

    # Quantity Sold Same Day Previous Week

    date_plus7_df = shop_date_level_features[["shop_id", "date", "shop_qty_sold_day"]]
    date_plus7_df["date_plus7"] = date_plus7_df["date"] + datetime.timedelta(days=7)

    date_plus7_df.drop(columns="date", inplace=True)
    date_plus7_df.rename(
        columns={"shop_qty_sold_day": "shop_qty_sold_last_dow", "date_plus7": "date"},
        inplace=True,
    )

    shop_date_level_features = shop_date_level_features.merge(
        date_plus7_df, on=["shop_id", "date"], how="left"
    )
    shop_date_level_features.shop_qty_sold_last_dow.fillna(0, inplace=True)

    # Number of Days in Previous 7-Day and 30-Day Periods with a Sale

    shop_date_level_features["day_w_sale"] = np.where(
        shop_date_level_features.shop_qty_sold_day > 0, 1, 0
    )

    shop_date_level_features[
        "shop_cnt_sale_dts_last_7d"
    ] = shop_date_level_features.groupby("shop_id")["day_w_sale"].apply(
        lambda x: x.rolling(7, min_periods=1).sum().shift().fillna(0)
    )
    shop_date_level_features[
        "shop_cnt_sale_dts_last_30d"
    ] = shop_date_level_features.groupby("shop_id")["day_w_sale"].apply(
        lambda x: x.rolling(30, min_periods=1).sum().shift().fillna(0)
    )

    shop_date_level_features.drop("day_w_sale", axis=1, inplace=True)

    # Rolling values
    shop_date_level_features[
        "shop_rolling_7d_max_qty"
    ] = shop_date_level_features.groupby("shop_id")["shop_qty_sold_day"].apply(
        lambda x: x.rolling(7, min_periods=1).max().shift().fillna(0)
    )
    shop_date_level_features[
        "shop_rolling_7d_min_qty"
    ] = shop_date_level_features.groupby("shop_id")["shop_qty_sold_day"].apply(
        lambda x: x.rolling(7, min_periods=1).min().shift().fillna(0)
    )

    return shop_date_level_features


# SHOP-ITEM-DATE-LEVEL FEATURES
def build_shop_item_date_lvl_features():
    """Build dataframe of shop-item-date-level features.

    Returns:
    --------
    shop_item_date_level_features : pandas dataframe
        Dataframe with each row representing a unique combination of shop_id, item_id, and date and columns containing shop-item-date-level features
    """

    # Total Quantity Sold by Shop-Item-Date

    # create column with quantity sold by shop-item-date
    shop_item_date_level_features = (
        sales.groupby(["shop_id", "item_id", "date"])["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "shop_item_qty_sold_day"})
    )

    # Add missing shop-item-dates (between first and last dates for each shop-item) with 0 values for quantity sold
    all_shop_item_dates = shop_item_date_level_features.groupby(
        ["shop_id", "item_id"]
    ).date.apply(drange)
    all_shop_item_dates = all_shop_item_dates.reset_index(level=(0, 1)).reset_index(
        drop=True
    )
    shop_item_date_level_features = all_shop_item_dates.merge(
        shop_item_date_level_features, on=["shop_id", "item_id", "date"], how="left"
    )
    shop_item_date_level_features.shop_item_qty_sold_day.fillna(0, inplace=True)

    # Add missing shop-item-dates (between last observed sale date and last day of the training period) for shop-items that exist in test dataset
    test_shop_items = (
        test_df[["shop_id", "item_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by=["shop_id", "item_id"])
    )
    last_shop_item_dts_in_train_data = (
        shop_item_date_level_features.groupby(["shop_id", "item_id"])
        .date.max()
        .reset_index()
        .rename(columns={"date": "last_shop_item_date"})
    )

    last_shop_item_dts_in_train_data["last_train_dt"] = FIRST_DAY_OF_TEST_PRD

    addl_dates = (
        last_shop_item_dts_in_train_data.set_index(["shop_id", "item_id"])
        .stack()
        .reset_index(level=2, drop=True)
        .to_frame()
        .rename(columns={0: "date"})
    )
    addl_dates = addl_dates.groupby(addl_dates.index).apply(
        lambda x: x.set_index("date").resample("D").asfreq()
    )
    addl_dates.reset_index(inplace=True)
    addl_dates["shop_id"], addl_dates["item_id"] = zip(*addl_dates.level_0)
    addl_dates.drop("level_0", axis=1, inplace=True)
    addl_dates = addl_dates[addl_dates.date != FIRST_DAY_OF_TEST_PRD]
    addl_dates = addl_dates.groupby("shop_id").apply(lambda group: group.iloc[1:,])
    addl_dates.reset_index(drop=True, inplace=True)

    addl_dates = addl_dates.merge(
        test_shop_items, on=["shop_id", "item_id"], how="inner"
    )

    shop_item_date_level_features = pd.concat(
        [shop_item_date_level_features, addl_dates], axis=0, ignore_index=True
    )
    shop_item_date_level_features.sort_values(
        by=["shop_id", "item_id", "date"], inplace=True
    )
    shop_date_level_features.shop_item_qty_sold_day.fillna(0, inplace=True)

    # Last Total Quantity Sold for Each Shop-Item-Date

    # create lag column for quantity sold, grouped by shop-item
    shifted = shop_item_date_level_features.groupby(
        ["shop_id", "item_id"]
    ).shop_item_qty_sold_day.shift()
    shop_item_date_level_features = shop_item_date_level_features.join(
        shifted.rename("shop_item_qty_sold_day_lag")
    )
    shop_item_date_level_features.shop_item_qty_sold_day_lag.fillna(0, inplace=True)

    # Time Elapsed Since Previous Sale of Same Item at Same Shop

    # create column for time elapsed since previous sale of same item at same shop
    shop_item_date_level_features["shop_item_last_date"] = np.where(
        shop_item_date_level_features.shop_item_qty_sold_day > 0,
        shop_item_date_level_features.date,
        None,
    )
    shop_item_date_level_features["shop_item_last_date"] = pd.to_datetime(
        shop_item_date_level_features.shop_item_last_date
    )
    shop_item_date_level_features[
        "shop_item_last_date"
    ] = shop_item_date_level_features.shop_item_last_date.fillna(method="ffill").shift()
    shop_item_date_level_features.loc[
        shop_item_date_level_features.groupby(["shop_id", "item_id"])[
            "shop_item_last_date"
        ]
        .head(1)
        .index,
        "shop_item_last_date",
    ] = pd.NaT
    shop_item_date_level_features[
        "shop_item_days_since_prev_sale"
    ] = shop_item_date_level_features.date.sub(
        shop_item_date_level_features.shop_item_last_date
    ).dt.days
    shop_item_date_level_features.shop_item_days_since_prev_sale.fillna(0, inplace=True)
    shop_item_date_level_features.drop("shop_item_last_date", axis=1, inplace=True)

    # Time Elapsed Since First Sale Date of Same Item at Same Shop

    # create column for time elapsed since first sale date of same item at same shop
    shop_item_date_level_features["shop_item_days_since_first_sale"] = (
        shop_item_date_level_features["date"]
        - shop_item_date_level_features.groupby(["shop_id", "item_id"])[
            "date"
        ].transform("first")
    ).dt.days

    # Indicator Columns for First Week and First Month of Sale of Item at Shop

    # create indicator column for first week of sale of item at shop
    shop_item_date_level_features["shop_item_first_week"] = (
        shop_item_date_level_features["shop_item_days_since_first_sale"] <= 6
    ).astype(np.int8)
    # create indicator column for first month of sale of item at shop
    shop_item_date_level_features["shop_item_first_month"] = (
        shop_item_date_level_features["shop_item_days_since_first_sale"] <= 30
    ).astype(np.int8)

    # Number of Days in Previous 7-Day and 30-Day Periods with a Sale

    shop_item_date_level_features["day_w_sale"] = np.where(
        shop_item_date_level_features.shop_item_qty_sold_day > 0, 1, 0
    )

    shop_item_date_level_features[
        "shop_item_cnt_sale_dts_last_7d"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "day_w_sale"
    ].apply(
        lambda x: x.rolling(7, min_periods=1).sum().shift().fillna(0)
    )
    shop_item_date_level_features[
        "shop_item_cnt_sale_dts_last_30d"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "day_w_sale"
    ].apply(
        lambda x: x.rolling(30, min_periods=1).sum().shift().fillna(0)
    )

    shop_item_date_level_features.drop("day_w_sale", axis=1, inplace=True)

    # Expanding Max, Min and Mean Quantity Values

    shop_item_date_level_features[
        "shop_item_expand_qty_max"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].apply(
        lambda x: x.expanding().max().shift().bfill()
    )

    shop_item_date_level_features[
        "shop_item_expand_qty_min"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].apply(
        lambda x: x.expanding().min().shift().bfill()
    )

    shop_item_date_level_features[
        "shop_item_expand_qty_mean"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].apply(
        lambda x: x.expanding().mean().shift().bfill()
    )

    # Quantity Sold 1, 2, 3 Days Ago

    shop_item_date_level_features[
        "shop_item_qty_sold_1d_ago"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].shift()
    shop_item_date_level_features.shop_item_qty_sold_1d_ago.fillna(0, inplace=True)

    shop_item_date_level_features[
        "shop_item_qty_sold_2d_ago"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].shift(
        2
    )
    shop_item_date_level_features.shop_item_qty_sold_2d_ago.fillna(0, inplace=True)

    shop_item_date_level_features[
        "shop_item_qty_sold_3d_ago"
    ] = shop_item_date_level_features.groupby(["shop_id", "item_id"])[
        "shop_item_qty_sold_day"
    ].shift(
        3
    )
    shop_item_date_level_features.shop_item_qty_sold_3d_ago.fillna(0, inplace=True)

    return shop_item_date_level_features


# Downcast Numeric Columns to Reduce Memory Usage
# from https://hackersandslackers.com/downcast-numerical-columns-python-pandas/
# with exclude=['uint32'] added to downcast_all function to avoid errors on existing unsigned integer columns


def float_to_int(ser):
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


def multi_assign(df, transform_fn, condition):
    df_to_use = df.copy()

    return df_to_use.assign(
        **{col: transform_fn(df_to_use[col]) for col in condition(df_to_use)}
    )


def all_float_to_int(df):
    df_to_use = df.copy()
    transform_fn = float_to_int
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)

    return multi_assign(df_to_use, transform_fn, condition)


def downcast_all(df, target_type, inital_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if inital_type is None:
        inital_type = target_type

    df_to_use = df.copy()

    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)

    condition = lambda x: list(
        x.select_dtypes(include=[inital_type], exclude=["uint32"]).columns
    )

    return multi_assign(df_to_use, transform_fn, condition)


def downcast(df_in):
    return (
        df_in.pipe(all_float_to_int)
        .pipe(downcast_all, "float")
        .pipe(downcast_all, "integer")
        .pipe(downcast_all, target_type="unsigned", inital_type="integer")
    )


def add_col_prefix(df, prefix, cols_not_to_rename=["shop_id", "item_id", "date"]):
    df.columns = [
        prefix + col
        if ((col not in cols_not_to_rename) & (~col.startswith(prefix)))
        else col
        for col in df.columns
    ]
    return df


def main():
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

    # next step
    sales = clean_sales_data()
    shops_df = build_shop_lvl_features()
    item_level_features = build_item_lvl_features()
    date_level_features = build_date_lvl_features()
    item_date_level_features = build_item_date_lvl_features()
    shop_date_level_features = build_shop_date_lvl_features()
    shop_item_date_level_features = build_shop_item_date_lvl_features()

    shops_df = downcast(shops_df)
    item_level_features = downcast(item_level_features)
    date_level_features = downcast(date_level_features)
    item_date_level_features = downcast(item_date_level_features)
    shop_date_level_features = downcast(shop_date_level_features)
    shop_item_date_level_features = downcast(shop_item_date_level_features)

    # MERGE ALL DATASETS INTO ONE

    # Rename columns in each dataset for easy identification by adding a prefix to column names

    shops_df = add_col_prefix(shops_df, "s_")
    item_level_features = add_col_prefix(item_level_features, "i_")
    date_level_features = add_col_prefix(date_level_features, "d_")
    item_date_level_features = add_col_prefix(item_date_level_features, "id_")
    shop_date_level_features = add_col_prefix(shop_date_level_features, "sd_")
    shop_item_date_level_features = add_col_prefix(
        shop_item_date_level_features, "sid_"
    )

    # Merge All Dataframes Into One

    all_features = shop_item_date_level_features.merge(
        shops_df, on="shop_id", how="inner"
    )
    all_features = all_features.merge(item_level_features, on="item_id", how="inner")
    all_features = all_features.merge(date_level_features, on="date", how="inner")
    all_features = all_features.merge(
        item_date_level_features, on=["item_id", "date"], how="inner"
    )
    all_features = all_features.merge(
        shop_date_level_features, on=["shop_id", "date"], how="inner"
    )

    all_features.to_pickle("../../../../../Volumes/My Passport/all_features_v3.pkl")
    # all_features.to_pickle('./Data/competitive-data-science-predict-future-sales/all_features.pkl')

    del all_features


if __name__ == "__main__":
    main()
