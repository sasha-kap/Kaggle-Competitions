"""
Functions:

run_query
- Run specified SQL query and export results in CSV format to specified
s3 bucket (or request execution plan of the query)

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    # query shop-item-dates for October 2015
    python3 postgres_to_s3.py 1510

    # request query execution plan only
    python3 postgres_to_s3.py 1510 -e

"""

import argparse
import csv
import datetime
from dateutil.relativedelta import relativedelta
import logging
from pathlib import Path
from pprint import pformat

import psycopg2
from psycopg2.sql import SQL

# Import the 'config' function from the config.py file
from config import config
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer


@Timer(logger=logging.info)
def run_query(sql_query, params=None, explain=False):
    """ Connect to the PostgreSQL database server and run specified query,
    exporting results in CSV format to specified s3 bucket.

    Parameters:
    -----------
    sql_query : str
        SQL query to execute
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method
    explain : bool
        Whether to request the query execution plan without actually
        executing the query

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
        logging.debug(f"SQL query to be executed: {sql_query}")

        if explain:
            # execute the provided SQL query
            cur.execute("EXPLAIN (FORMAT JSON) " + sql_query)
            logging.debug(pformat(cur.fetchall()))

            # close the communication with the PostgreSQL
            cur.close()

        else:
            if isinstance(sql_query, str):
                cur.execute(sql_query, params)

                # Make the changes to the database persistent
                conn.commit()
                # close the communication with the PostgreSQL
                cur.close()
            elif isinstance(sql_query, list):
                for q in sql_query:
                    cur.execute(q, params)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yrmonth",
        metavar="<yrmonth>",
        help="year-month for which to export data to CSV (required format: yymm)",
    )
    parser.add_argument(
        "--explain_plan",
        "-e",
        default=False,
        action="store_true",
        help="provides execution plan without executing the query (if included) or not (if not included)",
    )
    args = parser.parse_args()

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
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    start_instance()

    col_name_dict = {
        "shop_item_dates": "sid",
        "shops": "s",
        "items": "i",
        "dates": "d",
        "shop_dates": "sd",
        "item_dates": "id",
        "shop_cat_dates": "scd",
        "sid_n_sale_dts": "nsd",
        "sid_expand_qty_cv_sqrd": "ecv",
        "sid_expand_qty_stats": "eqs",
        "sid_roll_qty_stats": "rqs",
        "sid_expand_bw_sales_stats": "ebw",
        "addl_shop_item_dates": "sid",
        "sid_addl_n_sale_dts": "nsd",
        "sid_addl_expand_qty_cv_sqrd": "ecv",
        "sid_addl_expand_qty_stats": "eqs",
        "sid_addl_roll_qty_stats": "rqs",
        "sid_addl_expand_bw_sales_stats": "ebw",
    }

    # table_name,pg_relation_size
    # shops,8192
    # dates,442368
    # items,4022272
    # shop_dates,7716864
    # test_data,9486336
    # sales_cleaned,153190400
    # shop_cat_dates,230793216
    # item_dates,1495867392
    # sid_addl_roll_qty_stats,3093823488
    # sid_addl_n_sale_dts,3093823488
    # sid_addl_expand_qty_cv_sqrd,3093823488
    # sid_addl_expand_qty_stats,3093823488
    # sid_addl_expand_bw_sales_stats,3645587456
    # sid_n_sale_dts,5469306880
    # sid_expand_qty_cv_sqrd,5469306880
    # sid_expand_qty_stats,6339379200
    # sid_roll_qty_stats,6339379200
    # sid_expand_bw_sales_stats,7155671040
    # addl_shop_item_dates,8592179200
    # shop_item_dates,30550417408

    sql_col_set = set()
    sql_col_list = list()
    # removed irrelevant "test" table rows from the CSV manually
    with open("./rds_db/columns_output_2021_08_18_16_50.csv", "r") as col_file:
        csv_reader = csv.reader(col_file, delimiter=",")
        next(csv_reader, None)  # skip header row
        for row in csv_reader:
            if row[1] != "sales_cleaned":
                # three columns are not added to column list because they will be dealt with
                # separately in the SQL query below
                # also, duplicate column names are removed from final column list with the help of the set
                if row[2] not in sql_col_set and row[2] not in [
                    "sid_shop_cat_qty_sold_day",
                    "sid_shop_cat_qty_sold_last_7d",
                    "sid_cat_sold_at_shop_before_day_flag",
                ]:
                    sql_col_set.add(row[2])
                    sql_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
            # if row[1] != 'sales_cleaned' and not (
            #     row[1] != 'shop_item_dates' and row[2] in ['shop_id', 'item_id', 'sale_date', 'sid_item_category_id']
            # ):
            #     sql_col_list.append(".".join([col_name_dict[row[1]], row[2]]))
    cols_to_select = ", ".join(sql_col_list)
    cols_to_select_addl = cols_to_select.replace('sid.', 'asid.')

    ymd = (
        datetime.datetime.strptime(args.yrmonth, "%y%m")
        .strftime("%Y,%m,%d")
        .replace(",0", ",")
    )
    ymd_end = (
        (datetime.datetime.strptime(args.yrmonth, "%y%m") + relativedelta(day=31))
        .strftime("%Y,%m,%d")
        .replace(",0", ",")
    )
    # this query will not work because same columns exist in multiple tables being joined
    # (e.g., sid_addl_expand_bw_sales_stats and sid_expand_bw_sales_stats)
    # query = (
    #     f"SELECT {cols_to_select}, "
    #     "CASE WHEN scd.sid_shop_cat_qty_sold_day IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_day END "
    #     "AS sid_shop_cat_qty_sold_day, "
    #     "CASE WHEN scd.sid_shop_cat_qty_sold_last_7d IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_last_7d END "
    #     "AS sid_shop_cat_qty_sold_last_7d, "
    #     "CASE WHEN scd.sid_cat_sold_at_shop_before_day_flag IS NULL THEN 0 ELSE scd.sid_cat_sold_at_shop_before_day_flag END "
    #     "AS sid_cat_sold_at_shop_before_day_flag "
    #     "FROM ("
    #         f"SELECT * FROM shop_item_dates WHERE sale_date >= make_date({ymd}) "
    #         f"AND sale_date <= make_date({ymd_end})"
    #         f"UNION ALL "
    #         f"SELECT * FROM addl_shop_item_dates WHERE sale_date >= make_date({ymd}) "
    #         f"AND sale_date <= make_date({ymd_end})"
    #     ") sid "
    #     "INNER JOIN shops s "
    #     "ON sid.shop_id = s.shop_id "
    #     "INNER JOIN items i "
    #     "ON sid.item_id = i.item_id "
    #     "INNER JOIN dates d "
    #     "ON sid.sale_date = d.sale_date "
    #     "INNER JOIN shop_dates sd "
    #     "ON sid.shop_id = sd.shop_id AND sid.sale_date = sd.sale_date "
    #     "INNER JOIN item_dates id "
    #     "ON sid.item_id = id.item_id AND sid.sale_date = id.sale_date "
    #     "LEFT JOIN shop_cat_dates scd "
    #     "ON sid.shop_id = scd.shop_id AND sid.sale_date = scd.sale_date "
    #     "AND sid.sid_item_category_id = scd.sid_item_category_id "
    #     "INNER JOIN sid_n_sale_dts nsd "
    #     "ON sid.shop_id = nsd.shop_id AND sid.item_id = nsd.item_id "
    #     "AND sid.sale_date = nsd.sale_date "
    #     "INNER JOIN sid_expand_qty_cv_sqrd ecv "
    #     "ON sid.shop_id = ecv.shop_id AND sid.item_id = ecv.item_id "
    #     "AND sid.sale_date = ecv.sale_date "
    #     "INNER JOIN sid_expand_qty_stats eqs "
    #     "ON sid.shop_id = eqs.shop_id AND sid.item_id = eqs.item_id "
    #     "AND sid.sale_date = eqs.sale_date "
    #     "INNER JOIN sid_roll_qty_stats rqs "
    #     "ON sid.shop_id = rqs.shop_id AND sid.item_id = rqs.item_id "
    #     "AND sid.sale_date = rqs.sale_date "
    #     "INNER JOIN sid_expand_bw_sales_stats ebw "
    #     "ON sid.shop_id = ebw.shop_id AND sid.item_id = ebw.item_id "
    #     "AND sid.sale_date = ebw.sale_date "
    #     "INNER JOIN sid_addl_n_sale_dts nsd "
    #     "ON sid.shop_id = nsd.shop_id AND sid.item_id = nsd.item_id "
    #     "AND sid.sale_date = nsd.sale_date "
    #     "INNER JOIN sid_addl_expand_qty_cv_sqrd ecv "
    #     "ON sid.shop_id = ecv.shop_id AND sid.item_id = ecv.item_id "
    #     "AND sid.sale_date = ecv.sale_date "
    #     "INNER JOIN sid_addl_expand_qty_stats eqs "
    #     "ON sid.shop_id = eqs.shop_id AND sid.item_id = eqs.item_id "
    #     "AND sid.sale_date = eqs.sale_date "
    #     "INNER JOIN sid_addl_roll_qty_stats rqs "
    #     "ON sid.shop_id = rqs.shop_id AND sid.item_id = rqs.item_id "
    #     "AND sid.sale_date = rqs.sale_date "
    #     "INNER JOIN sid_addl_expand_bw_sales_stats ebw "
    #     "ON sid.shop_id = ebw.shop_id AND sid.item_id = ebw.item_id "
    #     "AND sid.sale_date = ebw.sale_date"
    # )
    # this query started but took several hours and connection was lost
    # query execution plan is here: logging_2021_08_19_17_11.log
    query = (
        # "WITH sid AS ("
        #     f"SELECT * FROM shop_item_dates WHERE sale_date >= make_date({ymd}) "
        #     f"AND sale_date <= make_date({ymd_end})"
        # "), "
        # "asid AS ("
        #     f"SELECT * FROM addl_shop_item_dates WHERE sale_date >= make_date({ymd}) "
        #     f"AND sale_date <= make_date({ymd_end})"
        # ") "
        f"SELECT {cols_to_select}, "
        "CASE WHEN scd.sid_shop_cat_qty_sold_day IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_day END "
        "AS sid_shop_cat_qty_sold_day, "
        "CASE WHEN scd.sid_shop_cat_qty_sold_last_7d IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_last_7d END "
        "AS sid_shop_cat_qty_sold_last_7d, "
        "CASE WHEN scd.sid_cat_sold_at_shop_before_day_flag IS NULL THEN 0 ELSE scd.sid_cat_sold_at_shop_before_day_flag END "
        "AS sid_cat_sold_at_shop_before_day_flag "
        "FROM sid "
        "INNER JOIN shops s "
        "ON sid.shop_id = s.shop_id "
        "INNER JOIN items i "
        "ON sid.item_id = i.item_id "
        "INNER JOIN dates d "
        "ON sid.sale_date = d.sale_date "
        "INNER JOIN shop_dates sd "
        "ON sid.shop_id = sd.shop_id AND sid.sale_date = sd.sale_date "
        "INNER JOIN item_dates id "
        "ON sid.item_id = id.item_id AND sid.sale_date = id.sale_date "
        "LEFT JOIN shop_cat_dates scd "
        "ON sid.shop_id = scd.shop_id AND sid.sale_date = scd.sale_date "
        "AND sid.sid_item_category_id = scd.sid_item_category_id "
        "INNER JOIN sid_n_sale_dts nsd "
        "ON sid.shop_id = nsd.shop_id AND sid.item_id = nsd.item_id "
        "AND sid.sale_date = nsd.sale_date "
        "INNER JOIN sid_expand_qty_cv_sqrd ecv "
        "ON sid.shop_id = ecv.shop_id AND sid.item_id = ecv.item_id "
        "AND sid.sale_date = ecv.sale_date "
        "INNER JOIN sid_expand_qty_stats eqs "
        "ON sid.shop_id = eqs.shop_id AND sid.item_id = eqs.item_id "
        "AND sid.sale_date = eqs.sale_date "
        "INNER JOIN sid_roll_qty_stats rqs "
        "ON sid.shop_id = rqs.shop_id AND sid.item_id = rqs.item_id "
        "AND sid.sale_date = rqs.sale_date "
        "INNER JOIN sid_expand_bw_sales_stats ebw "
        "ON sid.shop_id = ebw.shop_id AND sid.item_id = ebw.item_id "
        "AND sid.sale_date = ebw.sale_date "
        "UNION ALL "
        f"SELECT {cols_to_select_addl}, "
        "CASE WHEN scd.sid_shop_cat_qty_sold_day IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_day END "
        "AS sid_shop_cat_qty_sold_day, "
        "CASE WHEN scd.sid_shop_cat_qty_sold_last_7d IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_last_7d END "
        "AS sid_shop_cat_qty_sold_last_7d, "
        "CASE WHEN scd.sid_cat_sold_at_shop_before_day_flag IS NULL THEN 0 ELSE scd.sid_cat_sold_at_shop_before_day_flag END "
        "AS sid_cat_sold_at_shop_before_day_flag "
        "FROM asid "
        "INNER JOIN shops s "
        "ON asid.shop_id = s.shop_id "
        "INNER JOIN items i "
        "ON asid.item_id = i.item_id "
        "INNER JOIN dates d "
        "ON asid.sale_date = d.sale_date "
        "INNER JOIN shop_dates sd "
        "ON asid.shop_id = sd.shop_id AND asid.sale_date = sd.sale_date "
        "INNER JOIN item_dates id "
        "ON asid.item_id = id.item_id AND asid.sale_date = id.sale_date "
        "LEFT JOIN shop_cat_dates scd "
        "ON asid.shop_id = scd.shop_id AND asid.sale_date = scd.sale_date "
        "AND asid.sid_item_category_id = scd.sid_item_category_id "
        "INNER JOIN sid_addl_n_sale_dts nsd "
        "ON asid.shop_id = nsd.shop_id AND asid.item_id = nsd.item_id "
        "AND asid.sale_date = nsd.sale_date "
        "INNER JOIN sid_addl_expand_qty_cv_sqrd ecv "
        "ON asid.shop_id = ecv.shop_id AND asid.item_id = ecv.item_id "
        "AND asid.sale_date = ecv.sale_date "
        "INNER JOIN sid_addl_expand_qty_stats eqs "
        "ON asid.shop_id = eqs.shop_id AND asid.item_id = eqs.item_id "
        "AND asid.sale_date = eqs.sale_date "
        "INNER JOIN sid_addl_roll_qty_stats rqs "
        "ON asid.shop_id = rqs.shop_id AND asid.item_id = rqs.item_id "
        "AND asid.sale_date = rqs.sale_date "
        "INNER JOIN sid_addl_expand_bw_sales_stats ebw "
        "ON asid.shop_id = ebw.shop_id AND asid.item_id = ebw.item_id "
        "AND asid.sale_date = ebw.sale_date"
    )
    # this query returned a considerably costlier execution plan than the query
    # above and was not tried
    # query execution plan is here: logging_2021_08_19_16_20.log
    # query = (
    #     "WITH sid AS ("
    #     f"SELECT * FROM shop_item_dates WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM addl_shop_item_dates WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     "), "
    #     "nsd AS ("
    #     f"SELECT * FROM sid_n_sale_dts WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM sid_addl_n_sale_dts WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     "), "
    #     "ecv AS ("
    #     f"SELECT * FROM sid_expand_qty_cv_sqrd WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM sid_addl_expand_qty_cv_sqrd WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     "), "
    #     "eqs AS ("
    #     f"SELECT * FROM sid_expand_qty_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM sid_addl_expand_qty_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     "), "
    #     "rqs AS ("
    #     f"SELECT * FROM sid_roll_qty_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM sid_addl_roll_qty_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     "), "
    #     "ebw AS ("
    #     f"SELECT * FROM sid_expand_bw_sales_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end}) "
    #     "UNION ALL "
    #     f"SELECT * FROM sid_addl_expand_bw_sales_stats WHERE sale_date >= make_date({ymd}) "
    #     f"AND sale_date <= make_date({ymd_end})"
    #     ") "
    #     f"SELECT {cols_to_select}, "
    #     "CASE WHEN scd.sid_shop_cat_qty_sold_day IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_day END "
    #     "AS sid_shop_cat_qty_sold_day, "
    #     "CASE WHEN scd.sid_shop_cat_qty_sold_last_7d IS NULL THEN 0 ELSE scd.sid_shop_cat_qty_sold_last_7d END "
    #     "AS sid_shop_cat_qty_sold_last_7d, "
    #     "CASE WHEN scd.sid_cat_sold_at_shop_before_day_flag IS NULL THEN 0 ELSE scd.sid_cat_sold_at_shop_before_day_flag END "
    #     "AS sid_cat_sold_at_shop_before_day_flag "
    #     "FROM sid "
    #     "INNER JOIN shops s "
    #     "ON sid.shop_id = s.shop_id "
    #     "INNER JOIN items i "
    #     "ON sid.item_id = i.item_id "
    #     "INNER JOIN dates d "
    #     "ON sid.sale_date = d.sale_date "
    #     "INNER JOIN shop_dates sd "
    #     "ON sid.shop_id = sd.shop_id AND sid.sale_date = sd.sale_date "
    #     "INNER JOIN item_dates id "
    #     "ON sid.item_id = id.item_id AND sid.sale_date = id.sale_date "
    #     "LEFT JOIN shop_cat_dates scd "
    #     "ON sid.shop_id = scd.shop_id AND sid.sale_date = scd.sale_date "
    #     "AND sid.sid_item_category_id = scd.sid_item_category_id "
    #     "INNER JOIN sid_n_sale_dts nsd "
    #     "ON sid.shop_id = nsd.shop_id AND sid.item_id = nsd.item_id "
    #     "AND sid.sale_date = nsd.sale_date "
    #     "INNER JOIN sid_expand_qty_cv_sqrd ecv "
    #     "ON sid.shop_id = ecv.shop_id AND sid.item_id = ecv.item_id "
    #     "AND sid.sale_date = ecv.sale_date "
    #     "INNER JOIN sid_expand_qty_stats eqs "
    #     "ON sid.shop_id = eqs.shop_id AND sid.item_id = eqs.item_id "
    #     "AND sid.sale_date = eqs.sale_date "
    #     "INNER JOIN sid_roll_qty_stats rqs "
    #     "ON sid.shop_id = rqs.shop_id AND sid.item_id = rqs.item_id "
    #     "AND sid.sale_date = rqs.sale_date "
    #     "INNER JOIN sid_expand_bw_sales_stats ebw "
    #     "ON sid.shop_id = ebw.shop_id AND sid.item_id = ebw.item_id "
    #     "AND sid.sale_date = ebw.sale_date"
    # )

    # "SELECT * from aws_s3.query_export_to_s3('select * from shops',"

    # creation of indexed tables instead of CTEs in attempt to improve
    # query performance
    initial_queries = (
        "DROP TABLE IF EXISTS sid;"
        "CREATE TABLE sid AS "
        f"SELECT * FROM shop_item_dates WHERE sale_date >= make_date({ymd}) "
        f"AND sale_date <= make_date({ymd_end});"
        "CREATE INDEX sid_sale_date_idx ON sid (sale_date);"
        "CREATE INDEX sid_shop_id_idx ON sid (shop_id);"
        "CREATE INDEX sid_item_id_idx ON sid (item_id);"
        "DROP TABLE IF EXISTS asid;"
        "CREATE TABLE asid AS "
        f"SELECT * FROM addl_shop_item_dates WHERE sale_date >= make_date({ymd}) "
        f"AND sale_date <= make_date({ymd_end});"
        "CREATE INDEX asid_sale_date_idx ON asid (sale_date);"
        "CREATE INDEX asid_shop_id_idx ON asid (shop_id);"
        "CREATE INDEX asid_item_id_idx ON asid (item_id)"
    ).split(";")
    if args.explain_plan:
        run_query(initial_queries)
        logging.info("Initial queries completed.")
        sql = query

    else:
        run_query(initial_queries)
        logging.info("Initial queries completed.")
        yy_mm = "_".join([args.yrmonth[:2], args.yrmonth[2:]])
        sql = (
            f"SELECT * from aws_s3.query_export_to_s3('{query}',"
            f"aws_commons.create_s3_uri('my-rds-exports', 'shops_{yy_mm}.csv', 'us-west-2'),"
            f"options :='format csv, header');"
        )
        # sql = SQL(
        #     f"SELECT * from aws_s3.query_export_to_s3('{query}',"
        #     f"aws_commons.create_s3_uri('my-rds-exports', 'shops.csv', 'us-west-2'),"
        #     f"options :='format csv');"
        # )
    run_query(sql, explain=args.explain_plan)


if __name__ == "__main__":
    main()
