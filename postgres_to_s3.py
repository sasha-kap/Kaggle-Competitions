"""
Functions:

run_query
- Run specified SQL query and export results in CSV format to specified
s3 bucket

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    python3 postgres_to_s3.py

"""

import csv
import datetime
import logging
from pathlib import Path

import psycopg2
from psycopg2.sql import SQL

# Import the 'config' function from the config.py file
from config import config
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer

@Timer(logger=logging.info)
def run_query(sql_query, params=None):
    """ Connect to the PostgreSQL database server and run specified query,
    exporting results in CSV format to specified s3 bucket.

    Parameters:
    -----------
    sql_query : str
        SQL query to execute
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method

    Returns:
    --------
    None
    """

    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")

        # connect to the PostgreSQL server
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_params)

        # create a cursor to perform database operations
        cur = conn.cursor()

        # execute the provided SQL query
        cur.execute(sql_query, params)

        # Make the changes to the database persistent
        conn.commit()
        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


def main():

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(lineno)-7d %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_dir = Path.cwd().joinpath("rds_db")
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    log_fname = f"logging_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
    log_path = log_dir.joinpath(log_fname)

    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format=fmt,
        datefmt=datefmt,
        filename=log_path,
    )

    # statements to suppress irrelevant logging by boto3-related libraries
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)

    start_instance()

    col_name_dict = {
        'shop_item_dates': 'sid',
        'shops': 's',
        'items': 'i',
        'dates': 'd',
        'shop_dates': 'sd',
        'item_dates': 'id',
        'shop_cat_dates': 'scd',
        'sid_n_sale_dts': 'nsd',
        'sid_expand_qty_cv_sqrd': 'ecv',
        'sid_expand_qty_stats': 'eqs',
        'sid_roll_qty_stats': 'rqs',
        'sid_expand_bw_sales_stats': 'ebw'
    }

    sql_col_list = []
    with open('./rds_db/columns_output_2021_05_05_13_34.csv', 'r') as col_file:
        csv_reader = csv.reader(col_file, delimiter=',')
        next(csv_reader, None) # skip header row
        for row in csv_reader:
            if row[1] != 'sales_cleaned' and not (
                row[1] != 'shop_item_dates' and row[2] in ['shop_id', 'item_id', 'sale_date']
            ):
                sql_col_list.append(".".join([col_name_dict[row[1]], row[2]]))

    cols_to_select = ", ".join(sql_col_list)

    query = (
        f"WITH sid AS ("
            f"SELECT * FROM shop_item_dates WHERE sale_date >= make_date(2015,8,1) "
            f"AND sale_date <= make_date(2015,8,31)"
        f") "
        f"SELECT {cols_to_select} FROM sid "
        f"LEFT JOIN shops s "
        f"ON sid.shop_id = s.shop_id "
        f"LEFT JOIN items i "
        f"ON sid.item_id = i.item_id "
        f"LEFT JOIN dates d "
        f"ON sid.sale_date = d.sale_date "
        f"LEFT JOIN shop_dates sd "
        f"ON sid.shop_id = sd.shop_id AND sid.sale_date = sd.sale_date "
        f"LEFT JOIN item_dates id "
        f"ON sid.item_id = id.item_id AND sid.sale_date = id.sale_date "
        f"LEFT JOIN shop_cat_dates scd "
        f"ON sid.shop_id = scd.shop_id AND sid.sale_date = scd.sale_date "
        f"AND sid.sid_item_category_id = scd.sid_item_category_id "
        f"LEFT JOIN sid_n_sale_dts nsd "
        f"ON sid.shop_id = nsd.shop_id AND sid.item_id = nsd.item_id "
        f"AND sid.sale_date = nsd.sale_date "
        f"LEFT JOIN sid_expand_qty_cv_sqrd ecv "
        f"ON sid.shop_id = ecv.shop_id AND sid.item_id = ecv.item_id "
        f"AND sid.sale_date = ecv.sale_date "
        f"LEFT JOIN sid_expand_qty_stats eqs "
        f"ON sid.shop_id = eqs.shop_id AND sid.item_id = eqs.item_id "
        f"AND sid.sale_date = eqs.sale_date "
        f"LEFT JOIN sid_roll_qty_stats rqs "
        f"ON sid.shop_id = rqs.shop_id AND sid.item_id = rqs.item_id "
        f"AND sid.sale_date = rqs.sale_date "
        f"LEFT JOIN sid_expand_bw_sales_stats ebw "
        f"ON sid.shop_id = ebw.shop_id AND sid.item_id = ebw.item_id "
        f"AND sid.sale_date = ebw.sale_date"
    )

    # "SELECT * from aws_s3.query_export_to_s3('select * from shops',"

    sql = (
        f"SELECT * from aws_s3.query_export_to_s3('{query}',"
        f"aws_commons.create_s3_uri('my-rds-exports', 'shops_15_08.csv', 'us-west-2'),"
        f"options :='format csv, header');"
    )
    # sql = SQL(
    #     f"SELECT * from aws_s3.query_export_to_s3('{query}',"
    #     f"aws_commons.create_s3_uri('my-rds-exports', 'shops.csv', 'us-west-2'),"
    #     f"options :='format csv');"
    # )
    run_query(sql)


if __name__ == "__main__":
    main()
