import csv
import logging
from io import StringIO

from sqlalchemy import create_engine
import psycopg2
from tqdm import tqdm

# Import the 'config' function from the config.py file
from config import config
from timer import Timer


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data
    (per https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
    and https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table)

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = "{}.{}".format(table.schema, table.name)
        else:
            table_name = table.name

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def chunker(seq, size):
    '''Create chunks (subsets) of specified size from input sequence.

    Parameters:
    -----------
    seq : sequence/collection (e.g., string, list, dictionary, dataframe)
        Sequence/collection that needs to be split into chunks
    size : int
        Size of individual chunks

    Returns:
    --------
    generator expression for chunks of specified size (the last chunk may be
    smaller than value of size parameter)
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


@Timer(logger=logging.info)
def write_df_to_sql(df, table_name, dtypes_dict):
    '''Write pandas DataFrame object to PostgreSQL table, while displaying
    progress bar. Rows will be written in chunks of up to 10,000 rows max.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe to be written to SQL table
    table_name : str
        Name of SQL table to write to
    dtypes_dict : dict
        Dictionary mapping df pandas data types to PostgreSQL data types

    Returns:
    --------
    None
    '''

    # Create dictionary of database configuration details
    db_details = config(section="postgresql")

    user = db_details["user"]
    passw = db_details["password"]
    host = db_details["host"]
    port = db_details["port"]
    dbase = db_details["database"]

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

    # write DF to SQL table with progress bar displayed
    # per https://stackoverflow.com/a/39495229/9987623
    chunksize = max(int(len(df) / 100), 10000)
    with tqdm(total=len(df)) as pbar:
        for i, cdf in enumerate(chunker(df, chunksize)):
            replace = "replace" if i == 0 else "append"
            cdf.to_sql(
                table_name,
                engine,
                if_exists=replace,
                index=False,
                method=psql_insert_copy,
                dtype = dtypes_dict
            )
            pbar.update(chunksize)
