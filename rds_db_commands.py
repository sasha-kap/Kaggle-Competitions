"""
Functions:

set_primary_key
- Connect to the PostgreSQL database and execute SQL query that sets primary
key(s) on specified table(s).

query_table_info
- Query existing AWS RDS PostgreSQL database to obtain information on tables and
columns

run_query
- Run specified SQL query and save results to CSV file

drop_tables
- Connect to the PostgreSQL database and execute SQL query that drops specified
tables.

get_table_size:
- Connect to the PostgreSQL database and query and log the size of specified
table.

create_db_table_from_query:
- Connect to the PostgreSQL database and execute SQL query that creates a new
database table.

df_from_sql_table
- Connect to the PostgreSQL database and export existing database table to
pandas dataframe.

df_from_sql_query
- Connect to the PostgreSQL database and write results of SQL query to pandas
DataFrame.

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    python3 rds_db_commands.py summary
    python rds_db_commands.py query --stop
    python rds_db_commands.py drop -t temp temp2

"""
import argparse
from collections import defaultdict
import datetime
import logging
from pathlib import Path
from pprint import pformat

import pandas as pd
import psycopg2
from psycopg2.sql import SQL, Identifier
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

# Import the 'config' function from the config.py file
from config import config
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer


@Timer(logger=logging.info)
def set_primary_key(table_col_dict=None, db_table_name=None, cols=None, conn=None):
    """Connect to the PostgreSQL database and execute SQL query that sets
    primary key(s) on specified table(s).

    Parameters:
    -----------
    table_col_dict : dict
        Dictionary of tables (keys) in which to set primary key(s) specified by
        column names (values), optional, default: None
    db_table_name : str
        Name of table in which to set primary key(s), optional, default: None
    cols : list
        List of column names to set as primary key(s), optional, default: None
    conn :

    Returns:
    --------
    None
    """
    if ((table_col_dict is None and db_table_name is None) or
        (table_col_dict is not None and db_table_name is not None)):
        raise TypeError(
            "Specify either dictionary of tables and column names, "
            "or individual table name as string, not both."
        )
    elif ((db_table_name is not None and cols is None) or
          (table_col_dict is not None and cols is not None)):
        raise TypeError(
            "Specify columns to be set as primary key either in dictionary of "
            "table and column names, or as separate argument as string or list."
        )
    try:
        # read connection parameters
        db_params = config(section="postgresql")
        # connect to the PostgreSQL server
        if conn is None:
            conn = psycopg2.connect(**db_params)
        # open a cursor to perform database operations
        # as context manager (see https://www.psycopg.org/docs/usage.html)
        with conn.cursor() as cur:
            if table_col_dict is not None:
                for tbl in table_col_dict:
                    sql = SQL("ALTER TABLE {0} ADD PRIMARY KEY ({1});").format(
                        Identifier(tbl),
                        SQL(", ").join([Identifier(col) for col in table_col_dict[tbl]])
                    )
                    cur.execute(sql)
            else:
                sql = SQL("ALTER TABLE {0} ADD PRIMARY KEY ({1});").format(
                    Identifier(db_table_name),
                    SQL(", ").join([Identifier(col) for col in cols])
                )
                cur.execute(sql)
            conn.commit()

        if table_col_dict is not None:
            for tbl in table_col_dict:
                logging.info(
                    f"The following primary keys were successfully added to {tbl}"
                    f": {', '.join(table_col_dict[tbl])}."
                )
        else:
            logging.info(
                f"The following primary keys were successfully added to {db_table_name}"
                f": {', '.join(cols)}."
            )

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")

# per https://www.postgresqltutorial.com/postgresql-python/connect/
@Timer(logger=logging.info)
def query_table_info(csv_dir):
    """ Connect to the PostgreSQL database server and query info on existing
    tables.

    Parameters:
    -----------
    csv_dir : str or pathlib.Path() object
        Directory in which to save CSV files with query results

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

        # execute a statement
        cur.execute("SELECT version()")

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        logging.info(f"PostgreSQL database version: {db_version}")

        # Show tables in PostgreSQL by querying data from the PostgreSQL catalog,
        # filtering out system tables
        # per https://www.postgresqltutorial.com/postgresql-show-tables/
        query = """
            SELECT *
            FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND
                  schemaname != 'information_schema'
        """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = (
            f"pg_tables_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        )
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

        # Get information on columns in all tables from the information_schema.columns catalog
        # https://www.postgresql.org/docs/current/infoschema-columns.html
        query = """
            SELECT
               table_schema,
               table_name,
               column_name,
               data_type,
               is_nullable
            FROM
               information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = (
            f"columns_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        )
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

        # Get information on colummn constraints (unique, primary key, foreign key)
        # per https://www.postgresql.org/docs/current/infoschema-key-column-usage.html
        query = """
            SELECT
               constraint_name,
               table_name,
               column_name,
               ordinal_position,
               position_in_unique_constraint
            FROM
               information_schema.key_column_usage
        """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = f"key_column_usage_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

        # Per https://stackoverflow.com/a/21738505/9987623
        query = """
            SELECT table_name, pg_relation_size(quote_ident(table_name))
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY 2
        """

        # Per https://stackoverflow.com/a/52706756/9987623
        # query = """
        #     SELECT current_database() AS database,
        #            pg_size_pretty(total_database_size) AS total_database_size,
        #            schema_name,
        #            table_name,
        #            pg_size_pretty(total_table_size) AS total_table_size,
        #            pg_size_pretty(table_size) AS table_size,
        #            pg_size_pretty(index_size) AS index_size
        #            FROM ( SELECT table_name,
        #                     table_schema AS schema_name,
        #                     pg_database_size(current_database()) AS total_database_size,
        #                     pg_total_relation_size(table_name) AS total_table_size,
        #                     pg_relation_size(table_name) AS table_size,
        #                     pg_indexes_size(table_name) AS index_size
        #                     FROM information_schema.tables
        #                     WHERE table_schema = 'public'
        #                     ORDER BY total_table_size
        #                 ) AS sizes
        # """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = f"table_size_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

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


@Timer(logger=logging.info)
def vacuum_analyze_db():
    """Connect to the PostgreSQL database server and run VACUUM ANALYZE on database
    to garbage-collect and collect statistics about the contents of tables in the database.

    Parameters:
    -----------
    None

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
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        # create a cursor to perform database operations
        cur = conn.cursor()

        query = (
            "VACUUM (VERBOSE, ANALYZE)"
        )

        logging.debug(f"SQL query to be executed: {query}")

        # execute the provided SQL query
        cur.execute(query)

        # Make the changes to the database persistent
        conn.commit()

# for line in pprint.pformat(ds).split('\n'):
#     logging.debug(line)
        for line in conn.notices:
            logging.debug(line.strip('\n'))

        # logging.debug(pformat(conn.notices, indent=10))
        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


@Timer(logger=logging.info)
def print_qry_exec_plan():
    """ Connect to the PostgreSQL database server and run EXPLAIN on a query.

    Parameters:
    -----------
    None

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

        query = (
            # first sale date for each item in train data
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM sales_cleaned "
            "GROUP BY item_id), "
            # first sale date for each shop in train data
            "sfd AS ("
            "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
            "FROM sales_cleaned "
            "GROUP BY shop_id), "
            # "not new" items (items in test data that are also in train data)
            "not_new_items AS ("
            "SELECT DISTINCT item_id "
            "FROM test_data "
            "WHERE item_id IN ("
            "SELECT DISTINCT item_id FROM sales_cleaned)), "
            # shop-items among "not new" items that exist in test but not in train data
            "sist AS ("
            "SELECT td.shop_id, td.item_id "
            "FROM test_data td "
            "INNER JOIN not_new_items nni "
            "ON td.item_id = nni.item_id "
            "LEFT JOIN sales_cleaned sc "
            "ON td.shop_id = sc.shop_id AND td.item_id = sc.item_id "
            "WHERE sc.shop_id IS NULL AND sc.item_id IS NULL) "
            # count total days between the latter of the first day the
            # item was sold anywhere during train period and the first day that the shop
            # sold any item during the train period to the end of the train period
            "SELECT sum(days_between_sale_dts) AS total_days_to_add "
            "FROM (SELECT sist.shop_id, sist.item_id, "
            "make_date(2015,11,1) - "
            "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS days_between_sale_dts "
            "FROM sist "
            "LEFT JOIN ifd "
            "ON sist.item_id = ifd.item_id "
            "LEFT JOIN sfd "
            "ON sist.shop_id = sfd.shop_id) sub"
        )

        logging.debug(f"SQL query to be executed: {query}")

        # execute the provided SQL query
        cur.execute("EXPLAIN (FORMAT JSON) " + query)
        logging.debug(pformat(cur.fetchall()))

        # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


@Timer(logger=logging.info)
def run_query(csv_dir):
    """ Connect to the PostgreSQL database server and run specified query.

    Parameters:
    -----------
    csv_dir : str or pathlib.Path() object
        Directory in which to save CSV files with query results

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

        # prepare a query
        # query = (
        #     "SELECT id_cat_qty_sold_last_7d, "
        #     "COUNT(id_cat_qty_sold_last_7d) + "
        #     "COUNT(CASE WHEN id_cat_qty_sold_last_7d "
        #     "IS NULL THEN 1 ELSE NULL END) as CountOf "
        #     "FROM item_dates "
        #     "GROUP BY id_cat_qty_sold_last_7d "
        #     "ORDER BY id_cat_qty_sold_last_7d"
        # )
        col_list = ["shop_id", "item_id", "date"]
        level = 'item_id'

        col_name = (
            f"num_unique_"
            f"{[col for col in col_list if col not in (level, 'date')][0].replace('_id','s')}"
            f"_prior_to_day"
        )

        sql_str = (
            "WITH cte AS ("
            "SELECT a.*, {0}, {1} "
            "FROM df_temp AS a "
            "LEFT JOIN group_dates_w_val_cts AS b "
            "ON {2} = {3} AND {5} = {6} "
            "LEFT JOIN daily_cts_wo_lag AS c "
            "ON {2} = {4} AND {5} = {7} "
            "WHERE {0} IS NULL OR {1} IS NULL) "
            "SELECT COUNT(*) FROM cte;"
        )
        query = (
            "WITH cte AS ("
            "SELECT a.*, b.num_unique_shops_prior_to_day, c._daily_cts_wo_lag "
            "FROM df_temp AS a "
            "LEFT JOIN group_dates_w_val_cts AS b "
            "ON a.item_id = b.item_id AND a.sale_date = b.sale_date "
            "LEFT JOIN daily_cts_wo_lag AS c "
            "ON a.item_id = c.item_id AND a.sale_date = c.sale_date "
            # "WHERE b.num_unique_shops_prior_to_day IS NULL or c._daily_cts_wo_lag IS NULL) "
            # "WHERE b.num_unique_shops_prior_to_day IS NULL) "
            "WHERE c._daily_cts_wo_lag IS NULL) "
            "SELECT COUNT(*) FROM CTE"
        )
        query = (
            "SELECT COUNT(*) FROM daily_cts_wo_lag"
        )
        query = (
            "WITH cte AS ("
            "SELECT a.*, b.num_unique_shops_prior_to_day, c._daily_cts_wo_lag "
            "FROM df_temp AS a "
            "LEFT JOIN group_dates_w_val_cts AS b "
            "ON a.item_id = b.item_id "
            "LEFT JOIN daily_cts_wo_lag AS c "
            "ON a.item_id = c.item_id "
            "WHERE b.num_unique_shops_prior_to_day IS NULL or c._daily_cts_wo_lag IS NULL) "
            # "WHERE b.num_unique_shops_prior_to_day IS NULL) "
            # "WHERE c._daily_cts_wo_lag IS NULL) "
            "SELECT COUNT(*) FROM CTE"
        )
        # check how many rows have a null value in any column
        query = (
            "SELECT COUNT(*) AS n_null_rows "
            "FROM (SELECT * FROM test_table WHERE NOT (test_table IS NOT NULL)) sub"
        )

        # query = SQL(sql_str).format(
        #     # 0: column from group_dates_w_val_cts
        #     Identifier("b", col_name),
        #     # 1: column from daily_cts_wo_lag
        #     Identifier("c", "_daily_cts_wo_lag"),
        #     # 2-7: columns to merge on
        #     Identifier("a", level),
        #     Identifier("b", level),
        #     Identifier("c", level),
        #     Identifier("a", "sale_date"),
        #     Identifier("b", "sale_date"),
        #     Identifier("c", "sale_date"),
        #     # 8-9: columns to sort by
        #     Identifier("a", level),
        #     Identifier("a", "sale_date"),
        # )

        query = (
            "SELECT COUNT(*) FROM sid_big_query_result"
        )

        query = (
            "SELECT * FROM "
            "(SELECT DISTINCT d_modern_education_share AS values, "
            "'d_modern_education_share' AS column "
            "FROM dates) a "
            "UNION ALL "
            "(SELECT DISTINCT d_old_education_build_share AS values, "
            "'d_old_education_build_share' AS column "
            "FROM dates) "
        )

        query = (
            "SELECT DISTINCT id_num_unique_shops_prior_to_day AS values, "
            "'id_num_unique_shops_prior_to_day' AS column "
            "FROM item_dates"
        )

        query = (
            "SELECT * FROM ("
            "(SELECT DISTINCT i_item_cat_grouped_by_game_console AS values, "
            "'i_item_cat_grouped_by_game_console' AS column "
            "FROM items) "
            "UNION ALL "
            "(SELECT DISTINCT i_item_category_broad AS values, "
            "'i_item_category_broad' AS column "
            "FROM items) "
            "UNION ALL "
            "(SELECT DISTINCT i_item_category_name AS values, "
            "'i_item_category_name' AS column "
            "FROM items)) a"
        )

        query = (
            "SELECT * FROM shops"
        )

        query = (
            "WITH sid AS ("
                "SELECT DISTINCT shop_id FROM shop_item_dates WHERE sale_date >= make_date(2015,10,1) "
                "AND sale_date <= make_date(2015,10,31)"
            ") "
            "SELECT sid.shop_id FROM sid "
            "LEFT JOIN shops s "
            "ON sid.shop_id = s.shop_id "
            "WHERE s.shop_id IS NULL"
        )

        query = (
            "SELECT DISTINCT id_num_unique_shops_prior_to_day AS unique_num_unique_shops_prior_to_day "
            "FROM item_dates"
        )

        query = (
            "SELECT item_id, count(sale_date) AS total_days FROM item_dates "
            "WHERE item_id in (SELECT item_id FROM item_dates WHERE id_item_qty_sold_day > 1) "
            "GROUP BY item_id"
        )

        query = (
            "SELECT item_id, sale_date, id_item_qty_sold_day, id_item_date_diff_bw_last_and_prev_qty "
            "FROM item_dates "
            "WHERE item_id in (9338, 21084, 22002) "
            "ORDER BY item_id, sale_date"
        )

        query = (
            "SELECT count(id) AS count_ids, "
            "count(shop_id) AS count_shops, count(item_id) AS count_items "
            "FROM test_data"
        )

        # shop-items among "not new" items that exist in test but not in train data
        query = (
            "WITH not_new_items AS ("
            "SELECT DISTINCT item_id "
            "FROM test_data "
            "WHERE item_id IN ("
            "SELECT DISTINCT item_id FROM sales_cleaned)) "
            "SELECT td.id, td.shop_id, td.item_id "
            "FROM test_data td "
            "INNER JOIN not_new_items nni "
            "ON td.item_id = nni.item_id "
            "LEFT JOIN sales_cleaned sc "
            "ON td.shop_id = sc.shop_id AND td.item_id = sc.item_id "
            "WHERE sc.shop_id IS NULL AND sc.item_id IS NULL "
            "ORDER BY shop_id, item_id"
        )

        query = (
            "SELECT DISTINCT item_id "
            "FROM test_data WHERE item_id NOT IN ("
            "SELECT DISTINCT item_id FROM sales_cleaned)"
        )

        # shop-items with their earliest sale date in train data in Feb 2013 or later
        query = (
            "SELECT shop_id, item_id, sid_item_category_id, sale_date, sid_shop_item_qty_sold_day "
            "FROM shop_item_dates "
            "WHERE (shop_id, item_id, sale_date) IN ("
            "SELECT shop_id, item_id, min(sale_date) "
            "FROM shop_item_dates "
            "GROUP BY shop_id, item_id "
            "HAVING min(sale_date) >= make_date(2013,2,1))"
        )

        query = (
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM shop_item_dates "
            "GROUP BY item_id), "
            "sifd AS ("
            "SELECT shop_id, item_id, min(sale_date) AS first_shop_item_sale_date "
            "FROM shop_item_dates "
            "GROUP BY shop_id, item_id) "
            "SELECT sum(days_between_sale_dts) AS total_days_to_add "
            "FROM (SELECT sifd.shop_id, sifd.item_id, "
            "sifd.first_shop_item_sale_date - ifd.first_item_sale_date AS days_between_sale_dts "
            "FROM sifd LEFT JOIN ifd "
            "ON sifd.item_id = ifd.item_id) t1"
        )
        # For each item in train data, compute the total number of days between the first date
        # that that item was sold anywhere (inclusive) and the first dates that the item was sold
        # at each other shop (exclusive)
        # - create table of item-first sale dates
        # - join that table with shop-item-level table of first sale dates using item_id and
        # calculate the difference in days between shop-item’s first sale date and item’s first sale date

        query = (
            "SELECT count(*) as total_rows FROM shop_item_dates"
        )

        # across all shop-items in train data,
        # count total days between the latter of the first day the item was sold anywhere
        # AND the first day that the shop sold any item (inclusive) to the day that’s
        # currently in the data when the first observed sale happens for that shop-item
        query = (
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM shop_item_dates "
            "GROUP BY item_id), "
            "sfd AS ("
            "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
            "FROM shop_item_dates "
            "GROUP BY shop_id), "
            "sifd AS ("
            "SELECT shop_id, item_id, min(sale_date) AS first_shop_item_sale_date "
            "FROM shop_item_dates "
            "GROUP BY shop_id, item_id) "
            "SELECT sum(days_between_sale_dts) AS total_days_to_add "
            "FROM (SELECT sifd.shop_id, sifd.item_id, "
            "sifd.first_shop_item_sale_date - "
            "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS days_between_sale_dts "
            "FROM sifd LEFT JOIN ifd "
            "ON sifd.item_id = ifd.item_id "
            "LEFT JOIN sfd "
            "ON sifd.shop_id = sfd.shop_id) t1"
        )

        # across all shop-items in test data but not in train data, limited to items
        # in train data, count total days between the latter of the first day the
        # item was sold anywhere during train period and the first day that the shop
        # sold any item during the train period to the end of the train period
        query = (
            # first sale date for each item in train data
            "WITH ifd AS ("
            "SELECT item_id, min(sale_date) AS first_item_sale_date "
            "FROM sales_cleaned "
            "GROUP BY item_id), "
            # first sale date for each shop in train data
            "sfd AS ("
            "SELECT shop_id, min(sale_date) AS first_shop_sale_date "
            "FROM sales_cleaned "
            "GROUP BY shop_id), "
            # "not new" items (items in test data that are also in train data)
            "not_new_items AS ("
            "SELECT DISTINCT item_id "
            "FROM test_data "
            "WHERE item_id IN ("
            "SELECT DISTINCT item_id FROM sales_cleaned)), "
            # shop-items among "not new" items that exist in test but not in train data
            "sist AS ("
            "SELECT td.shop_id, td.item_id "
            "FROM test_data td "
            "INNER JOIN not_new_items nni "
            "ON td.item_id = nni.item_id "
            "LEFT JOIN sales_cleaned sc "
            "ON td.shop_id = sc.shop_id AND td.item_id = sc.item_id "
            "WHERE sc.shop_id IS NULL AND sc.item_id IS NULL) "
            # count total days between the latter of the first day the
            # item was sold anywhere during train period and the first day that the shop
            # sold any item during the train period to the end of the train period
            "SELECT sum(days_between_sale_dts) AS total_days_to_add "
            "FROM (SELECT sist.shop_id, sist.item_id, "
            "make_date(2015,11,1) - "
            "GREATEST(sfd.first_shop_sale_date, ifd.first_item_sale_date) AS days_between_sale_dts "
            "FROM sist "
            "LEFT JOIN ifd "
            "ON sist.item_id = ifd.item_id "
            "LEFT JOIN sfd "
            "ON sist.shop_id = sfd.shop_id) sub"
        )

        logging.debug(f"SQL query to be executed: {query}")

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = (
            f"query_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        )
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, "w") as f:
            cur.copy_expert(outputquery, f)

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


@Timer(logger=logging.info)
def drop_tables(delete_tables, conn=None):
    """Connect to the PostgreSQL database and execute SQL query that drops
    specified tables.

    Parameters:
    -----------
    delete_tables : list
        List of tables to delete from PostgreSQL database
    conn :

    Returns:
    --------
    None
    """
    try:
        # read connection parameters
        db_params = config(section="postgresql")
        # connect to the PostgreSQL server
        if conn is None:
            conn = psycopg2.connect(**db_params)
        # open a cursor to perform database operations
        # as context manager (see https://www.psycopg.org/docs/usage.html)
        with conn.cursor() as cur:
            sql = SQL("DROP TABLE IF EXISTS {0};").format(
                SQL(", ").join([Identifier(table) for table in delete_tables])
            )
            cur.execute(sql)
            conn.commit()

        logging.info(
            f"The following tables were successfully deleted "
            f"from DB: {', '.join(delete_tables)}"
        )
    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


def get_table_size(table_name):
    """Connect to the PostgreSQL database and query and log the size of
    specified table.

    Parameters:
    -----------
    table_name : str
        Name of table for which size needs to be queried

    Returns:
    --------
    None
    """
    # Create dictionary of database configuration details
    db_details = config(section="postgresql")

    user = db_details["user"]
    passw = db_details["password"]
    host = db_details["host"]
    port = db_details["port"]
    dbase = db_details["database"]

    conn = None
    try:
        engine = create_engine(
            "postgresql+psycopg2://"
            + user
            + ":"
            + passw
            + "@"
            + host
            + ":"
            + str(port)
            + "/"
            + dbase
        )

        sql = text(
            "SELECT table_name, pg_relation_size(quote_ident(table_name)) "
                "FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :name"
        )
        params = {"name": table_name}
        # Acquire a database connection
        conn = engine.connect()
        # Log size of specified table
        logging.info(f"Created {table_name} table's size is: {conn.execute(sql, params).fetchall()[0][1]:,}")

    except (Exception, SQLAlchemyError) as error:
        logging.exception(f"Table size query on {table_name} table was not executed.")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


@Timer(logger=logging.info)
def create_db_table_from_query(sql_query, params=None):
    """Connect to the PostgreSQL database and execute SQL query that creates
    a new database table.

    Parameters:
    -----------
    sql_query : SQLAlchemy Selectable (select or text object)
        SQL query to be executed
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
        # query and log size of newly created table
        get_table_size(params["db_table"])


val_err_dict = defaultdict(int)

def _cast_by_col(df, pd_types_dict):
    """Cast each column in input dataframe to specified data type.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe with columns that need to be cast to new data types
    pd_types_dict : dict
        Dictionary of dataframe data types to be used to cast table columns

    Returns:
    --------
    pandas DataFrame
    """
    for col in df.columns:
        try:
            df[col] = df[col].astype(pd_types_dict[col])
        # if column does not exist in dictionary, leave it as is
        except KeyError:
            pass
        # if null (or some other) values appear that cannot be converted to int type
        # keep a counter of how many chunks and which columns ran into this problem
        except ValueError:
            global val_err_dict
            val_err_dict[col] += 1
    return df


def df_from_sql_table(tbl, pd_types, date_list=None, delete_tables=None):
    """Connect to the PostgreSQL database and export existing database table
    to pandas dataframe.

    Parameters:
    -----------
    tbl : str
        Name of table to export to dataframe
    pd_types : dict
        Dictionary of dataframe data types to be used to cast table columns
    date_list : list or dict, optional, default: None
        List of column names to parse as dates
    delete_tables : list, optional, default: None
        List of tables to delete from PostgreSQL after exporting

    Returns:
    --------
    pandas DataFrame

    Note:
    -----
    This function currently does not handle deleting tables from database
    following creation of pandas dataframe, because of different conn object
    creation procedures in this and drop_tables functions. The drop_tables
    function is to be called separately following the call to this function,
    if needed.
    """
    # Create dictionary of database configuration details
    db_details = config(section="postgresql")

    user = db_details["user"]
    passw = db_details["password"]
    host = db_details["host"]
    port = db_details["port"]
    dbase = db_details["database"]

    conn = None
    try:
        engine = create_engine(
            "postgresql+psycopg2://"
            + user
            + ":"
            + passw
            + "@"
            + host
            + ":"
            + str(port)
            + "/"
            + dbase
        )
        conn = engine.connect().execution_options(stream_results=True)

        chunk_list = [
            _cast_by_col(chunk, pd_types) for chunk in pd.read_sql_table(
                tbl, conn, parse_dates=date_list, chunksize=10000
            )
        ]

        df = pd.concat(
            chunk_list,
            ignore_index=True,
        )

        return df

    except (Exception, SQLAlchemyError) as error:
        logging.exception("Exception occurred")

    finally:
        logging.debug(f"val_err_dict at the end of df_from_sql_table() is {val_err_dict}")
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


@Timer(logger=logging.info)
def df_from_sql_query(sql_query, pd_types, params=None, date_list=None, delete_tables=None):
    """Connect to the PostgreSQL database, execute SQL query and return
    results as pandas DataFrame.

    Parameters:
    -----------
    sql_query : SQLAlchemy Selectable (select or text object)
        SQL query to be executed
    pd_types : dict
        Dictionary of dataframe data types to be used to cast output of SQL query
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method
    date_list : list or dict, optional, default: None
        List of column names to parse as dates
    delete_tables : list, optional, default: None
        List of tables to delete from PostgreSQL after executing the SQL query

    Returns:
    --------
    pandas DataFrame

    """
    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**db_params)
        logging.debug(f"SQL query to be executed by read_sql_query(): {sql_query.as_string(conn)}")
        df = pd.concat(
            [
                # chunk.astype({k: v for k, v in pd_types.items() if k in chunk.columns}, errors='ignore')
                # chunk.astype({k: v for k, v in pd_types.items() if k in chunk.columns})
                _cast_by_col(chunk, pd_types) for chunk in pd.read_sql_query(
                    sql_query, conn, params=params, parse_dates=date_list, chunksize=10000
                )
            ],
            ignore_index=True,
        )
        return df

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    else:
        if delete_tables is not None:
            drop_tables(delete_tables, conn)

    finally:
        logging.debug(f"val_err_dict at the end of df_from_sql_query() is {val_err_dict}")
        if not conn.closed:
            conn.close()
            logging.info("Database connection closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'summary', 'primary', 'analyze', 'explain', 'query' or 'drop'",
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after querying (if included) or not (if not included)",
    )
    parser.add_argument(
        '-t',
        '--tables-list',
        nargs='+',
        default=[],
        help="list of tables to drop from database"
    )

    args = parser.parse_args()

    if args.command not in [
        "summary",
        "primary",
        "analyze",
        "explain",
        "query",
        "drop",
    ]:
        print(
            "'{}' is not recognized. "
            "Use 'summary', 'primary', 'analyze', 'explain', 'query', or 'drop'".format(args.command)
        )

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

    if args.command == "summary":
        query_table_info(log_dir)
    if args.command == "primary":
        # set_primary_key(db_table_name='shops', cols=['shop_id'])
        # set_primary_key(db_table_name='items', cols=['item_id'])
        # set_primary_key(db_table_name='shop_dates', cols=['shop_id','sale_date'])
        set_primary_key(table_col_dict={
            # 'sid_n_sale_dts' : ['shop_id', 'item_id', 'sale_date'],

            # 'sid_roll_qty_stats' : ['shop_id', 'item_id', 'sale_date'],
            # 'sid_expand_qty_cv_sqrd' : ['shop_id', 'item_id', 'sale_date'],
            # 'sid_expand_qty_stats' : ['shop_id', 'item_id', 'sale_date'],
            # 'sid_expand_bw_sales_stats' : ['shop_id', 'item_id', 'sale_date'],
            # 'shop_cat_dates' : ['shop_id', 'sid_item_category_id', 'sale_date'],
            # 'item_dates' : ['item_id', 'sale_date'],
            # 'shop_item_dates' : ['shop_id', 'item_id', 'sale_date'],
            'test_data': ['shop_id', 'item_id'],
            'sales_cleaned': ['shop_id', 'item_id', 'sale_date']

            # 'dates' : ['sale_date'],
        })

    elif args.command == "analyze":
        vacuum_analyze_db()
    elif args.command == "explain":
        print_qry_exec_plan()
    elif args.command == "query":
        run_query(log_dir)
    elif args.command == "drop":
        drop_tables(args.tables_list)

    if args.stop == True:
        stop_instance()

if __name__ == "__main__":
    main()
