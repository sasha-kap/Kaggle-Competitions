"""
Script to create, stop, or to start an RDS instance

Usage: run from the command line as such:

    # Create RDS instance
    python3 rds_instance_mgmt.py create

    # Stop RDS instance
    python3 rds_instance_mgmt.py stop

    # Start RDS instance
    python3 rds_instance_mgmt.py stop
"""

import time

import boto3
import botocore
from botocore.exceptions import ClientError

# Import the 'config' function from the config.py file
from config import config

def create_instance():
    '''Create RDS instance with PostgreSQL database and specified configuration.'''

    # Create dictionaries of instance and database configuration details
    rds_instance = config(section="rds")
    db_details = config(section="postgresql")

    rds = boto3.client("rds", region_name=rds_instance["region"])
    try:
        rds.create_db_instance(
            DBInstanceIdentifier=rds_instance["instance_id"],
            DBInstanceClass=rds_instance["instance_class"],
            AllocatedStorage=int(rds_instance["storage_gb"]),
            DBName=db_details["database"],
            Engine="postgres",
            EngineVersion=rds_instance["version"],
            AvailabilityZone=rds_instance["azone"],
            StorageType=rds_instance["storage_type"],
            StorageEncrypted=True,
            MultiAZ=False,
            MasterUsername=db_details["user"],
            MasterUserPassword=db_details["password"],
            # disable automatic backups
            BackupRetentionPeriod=0,
        )
        print("Starting RDS instance with ID: {}".format(rds_instance["instance_id"]))
    except ClientError as e:
        if e.response['Error']['Code'] == "DBInstanceAlreadyExists":
            print(
                "DB instance {} exists already, continuing to poll ...".format(
                    rds_instance["instance_id"]
                )
            )
        else:
            raise

    running = True
    while running:
        response = rds.describe_db_instances(
            DBInstanceIdentifier=rds_instance["instance_id"]
        )

        db_instances = response["DBInstances"]
        if len(db_instances) != 1:
            raise Exception(
                "Whoa cowboy! More than one DB instance returned; this should never happen"
            )

        db_instance = db_instances[0]

        status = db_instance["DBInstanceStatus"]

        print("Last DB status: {}".format(status))

        time.sleep(10)
        if status == "available":
            endpoint = db_instance["Endpoint"]
            host = endpoint["Address"]

            print("DB instance ready with host: {}".format(host))
            running = False


def stop_instance():
    '''Stop active RDS instance.'''

    rds_instance = config(section="rds")
    rds = boto3.client("rds", region_name=rds_instance["region"])
    try:
        response = rds.stop_db_instance(DBInstanceIdentifier=rds_instance["instance_id"])
    except ClientError as e:
        print(e)

    stopping = True
    while stopping:
        response = rds.describe_db_instances(
            DBInstanceIdentifier=rds_instance["instance_id"]
        )

        db_instances = response["DBInstances"]
        if len(db_instances) != 1:
            raise Exception(
                "Whoa cowboy! More than one DB instance returned; this should never happen"
            )

        db_instance = db_instances[0]

        status = db_instance["DBInstanceStatus"]

        print("Last DB status: {}".format(status))

        time.sleep(10)
        if status == "stopped":
            print("Instance stopped successfully")
            stopping = False

def start_instance():
    '''Start previously stopped RDS instance.'''

    rds_instance = config(section="rds")
    rds = boto3.client("rds", region_name=rds_instance["region"])
    try:
        response = rds.start_db_instance(DBInstanceIdentifier=rds_instance["instance_id"])
    except ClientError as e:
        print(e)

    starting = True
    while starting:
        response = rds.describe_db_instances(
            DBInstanceIdentifier=rds_instance["instance_id"]
        )

        db_instances = response["DBInstances"]
        if len(db_instances) != 1:
            raise Exception(
                "Whoa cowboy! More than one DB instance returned; this should never happen"
            )

        db_instance = db_instances[0]

        status = db_instance["DBInstanceStatus"]

        print("Last DB status: {}".format(status))

        time.sleep(10)
        if status == "available":
            endpoint = db_instance["Endpoint"]
            host = endpoint["Address"]

            print("DB instance ready with host: {}".format(host))
            starting = False



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'create', 'start' or 'stop'",
    )
    args = parser.parse_args()

    if args.command == "create":
        create_instance()
    elif args.command == "start":
        start_instance()
    elif args.command == "stop":
        stop_instance()
    else:
        print(
            "'{}' is not recognized. "
            "Use 'create', 'start' or 'stop'".format(args.command)
        )
