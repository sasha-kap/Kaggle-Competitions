"""
Functions:
1) Query existing AWS RDS PostgreSQL database to obtain information on tables and
columns
2) Run specified SQL query and save results to CSV file
2) Write results from SQL query to pandas DataFrame

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    python3 rds_db_commands.py summary
    python rds_db_commands.py query --stop

"""
import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.sql import SQL, Identifier

# Import the 'config' function from the config.py file
from config import config
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer

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
        query = (
            "SELECT id_cat_qty_sold_last_7d, "
            "COUNT(id_cat_qty_sold_last_7d) + "
            "COUNT(CASE WHEN id_cat_qty_sold_last_7d "
            "IS NULL THEN 1 ELSE NULL END) as CountOf "
            "FROM item_dates "
            "GROUP BY id_cat_qty_sold_last_7d "
            "ORDER BY id_cat_qty_sold_last_7d"
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
                chunk.astype({k: v for k, v in pd_types.items() if k in chunk.columns})
                for chunk in pd.read_sql_query(
                    sql_query, conn, params=params, parse_dates=date_list, chunksize=10000
                )
            ],
            ignore_index=True,
        )
        if delete_tables is not None:
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
        return df

    except (Exception, psycopg2.DatabaseError) as error:
        logging.exception("Exception occurred")

    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'summary' or 'query'",
    )
    parser.add_argument(
        "--stop",
        default=False,
        action="store_true",
        help="stop RDS instance after querying (if included) or not (if not included)",
    )

    args = parser.parse_args()

    if args.command not in [
        "summary",
        "query",
    ]:
        print(
            "'{}' is not recognized. "
            "Use 'summary' or 'query'".format(args.command)
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
    elif args.command == "query":
        run_query(log_dir)

    if args.stop == True:
        stop_instance()

if __name__ == "__main__":
    main()
