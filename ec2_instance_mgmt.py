"""
Script to create EC2 instance and associate the necessary IAM role with the
instance.

Usage: run from the command line as such:

    # Create EC2 instance
    python3 ec2_instance_mgmt.py start --dry_run
"""

import time

import boto3
import botocore
from botocore.exceptions import ClientError

# Import the 'config' function from the config.py file
from config import config

def start_instance(dry_run = False):

    # Create dictionary of instance configuration details
    ec2_instance = config(section="ec2")

    ec2 = boto3.resource('ec2', region_name=ec2_instance["region"])
    ec2_client = boto3.client('ec2', region_name=ec2_instance["region"])

    try:
        # not using user_data script as it is run as root user,
        # which does not work for creating a virtual environment for ec2-user
        # user_data = open('ec2_user_data.txt').read()

        instance = ec2.create_instances(
            ImageId = ec2_instance["image_id"],
            InstanceType = ec2_instance["instance_type"],
            Placement={
                'AvailabilityZone': ec2_instance["azone"],
            },
            SecurityGroupIds=[ec2_instance["sec_grp"]],
            KeyName = ec2_instance["key_pair"],
            MaxCount=1,
            MinCount=1,
            IamInstanceProfile={'Name': ec2_instance["iam_profile"]},
            # UserData=user_data,
            DryRun=dry_run
        )
        print("Starting EC2 {0} instance in {1} availability zone".format(ec2_instance["instance_type"], ec2_instance["azone"]))
    except ClientError as e:
        if e.response['Error']['Code'] == "DryRunOperation":
            print(
                "Dry run successful - request would have succeeded"
                )
        else:
            raise

    if not dry_run:
        starting = True
        while starting:
            time.sleep(3)

            response = ec2_client.describe_instance_status(
                IncludeAllInstances=True
            )
            ec2_instances = response["InstanceStatuses"]

            # filter list of instance dictionaries to instances with running state
            running_instances = [inst for inst in ec2_instances if inst["InstanceState"]["Name"] == 'running']
            if len(running_instances) > 1:
                raise Exception(
                    "Whoa cowboy! More than one EC2 instance in running state; this should not happen"
                )

            if running_instances:
                # create instance_id string
                instance_id = instance[0].id

                instance = ec2.Instance(instance_id)
                print("Instance id: ", instance.id)
                print("Instance public IP: ", instance.public_ip_address)
                print("Instance private IP: ", instance.private_ip_address)
                print("Public dns name: ", instance.public_dns_name)

                # iam_request = ec2_client.associate_iam_instance_profile(
                #     IamInstanceProfile={
                #         # 'Arn': 'string',
                #         'Name': iam_profile
                #     },
                #     InstanceId=instance.id
                # )
                #
                # time.sleep(5)
                # assoc_id = iam_request['IamInstanceProfileAssociation']['AssociationId']
                # associating_iam = True
                #
                # while associating_iam:
                #     time.sleep(2)
                #     iam_response = ec2_client.describe_iam_instance_profile_associations(
                #         AssociationIds=[
                #             assoc_id,
                #         ],
                #     )
                #     if iam_response['IamInstanceProfileAssociations']['State'] == 'associated':
                #         print(f"IAM profile {iam_profile} successfully associated with instance")
                #         associating_iam = False

                starting = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'start'",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="dry run of instance creation (if included) or not (if not included)",
    )
    args = parser.parse_args()

    if args.command == "start":
        start_instance(dry_run=args.dry_run)
    else:
        print(
            "'{}' is not recognized. "
            "Use 'start'".format(args.command)
        )
