"""
1. Read S3 CSV files
2. Iterate over CSV chunks and apply StandardScaler (two passes: fit and transform)
(per https://stackoverflow.com/questions/52642940/feature-scaling-for-a-big-dataset)
3. Pass scaled data to IncrementalPCA() (partial_fit, then transform on the third pass
over CSV chunks)
"""

from itertools import chain, tee

import boto3
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

bucket = "my-rds-exports"
# file_name1 = "csv1.csv"
# file_name2 = "csv2.csv"

s3_client = boto3.client("s3")

csv_list = [
    key["Key"]
    for key in s3_client.list_objects(Bucket="my-rds-exports", Prefix="csv")["Contents"]
]

# response1 = s3_client.get_object(Bucket=bucket, Key=file_name1)
# status1 = response1.get("ResponseMetadata", {}).get("HTTPStatusCode")
#
# response2 = s3_client.get_object(Bucket=bucket, Key=file_name2)
# status2 = response2.get("ResponseMetadata", {}).get("HTTPStatusCode")

# if status1 == 200 and status2 == 200:
#     print(f"Successful S3 get_object response. Status1 - {status1}. Status2 - {status2}.")
# reader1, reader2, reader3 = tee(chain(pd.read_csv(response1.get("Body"), chunksize=100),
#     pd.read_csv(response2.get("Body"), chunksize=100)), 3)

reader1, reader2, reader3 = tee(
    chain.from_iterable(
        [
            pd.read_csv(
                s3_client.get_object(Bucket=bucket, Key=csv_file).get("Body"),
                chunksize=100,
            )
            for csv_file in csv_list
        ]
    ),
    3,
)
scaler = StandardScaler()
sklearn_pca = IncrementalPCA(n_components=1)
for chunk in reader1:
    # scaler.partial_fit(chunk.sample(frac=0.5, random_state=42)['x'].to_numpy().reshape(1,-1))
    scaler.partial_fit(chunk.sample(frac=0.5, random_state=42)[["x", "y"]])
    # print(scaler.mean_, scaler.var_)

all_scaled_data = []
# reader = pd.read_csv(response.get("Body"), chunksize=100)
for chunk in reader2:
    scaled_data = scaler.transform(chunk.sample(frac=0.5, random_state=42)[["x", "y"]])
    # add each chunk of transformed data to list
    all_scaled_data.append(scaled_data)
    sklearn_pca.partial_fit(scaled_data)
print(len(all_scaled_data))
# print(type(all_scaled_data))
print(all_scaled_data[0].shape)

pca_transformed = None
for chunk in reader3:
    tx_chunk = sklearn_pca.transform(
        chunk.sample(frac=0.5, random_state=42)[["x", "y"]]
    )
    if pca_transformed is None:
        pca_transformed = tx_chunk
    else:
        pca_transformed = np.vstack((pca_transformed, tx_chunk))

print(pca_transformed.shape)
print(sklearn_pca.explained_variance_, sklearn_pca.explained_variance_ratio_)

# else:
#     print(f"Unsuccessful S3 get_object response. Status1 - {status1}. Status2 - {status2}.")
