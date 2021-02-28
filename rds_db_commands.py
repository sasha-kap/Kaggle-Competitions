"""
Query existing AWS RDS PostgreSQL database to obtain information on tables and
columns.

Copyright (c) 2021 Sasha Kapralov
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: run from the command line as such:

    python3 rds_db_commands.py

"""
import datetime
import logging
from pathlib import Path

import psycopg2

# Import the 'config' function from the config.py file
from config import config
from rds_instance_mgmt import start_instance, stop_instance
from timer import Timer

# per https://www.postgresqltutorial.com/postgresql-python/connect/
@Timer(logger=logging.info)
def query_table_info(csv_dir):
    """ Connect to the PostgreSQL database server and query info on existing
    tables.
    """
    conn = None
    try:
        # read connection parameters
        db_params = config(section="postgresql")

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**db_params)

        # create a cursor to perform database operations
        cur = conn.cursor()

	    # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

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
        csv_fname = f"pg_tables_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, 'w') as f:
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
        csv_fname = f"columns_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, 'w') as f:
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
        with open(csv_path, 'w') as f:
            cur.copy_expert(outputquery, f)

        # Per https://stackoverflow.com/a/21738505/9987623
        query = """
            SELECT table_name, pg_relation_size(quote_ident(table_name))
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY 2
        """

        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        csv_fname = f"table_size_output_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        csv_path = csv_dir.joinpath(csv_fname)
        with open(csv_path, 'w') as f:
            cur.copy_expert(outputquery, f)

	    # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def main():

    fmt = "%(name)-12s : %(asctime)s %(levelname)-8s %(message)s"
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

    logger = logging.getLogger()

    start_instance()
    query_table_info(log_dir)
    stop_instance()

if __name__ == '__main__':
    main()
