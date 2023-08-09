# Databricks notebook source
# ADLS Gen 2 mounting
blob_container  = "w261datastore"
storage_account = "w261adlsgen2"
secret_scope    = "w261-keyvault"                   
secret_key      = "w261-storage-key"                     
NAME: str = "TP"
# team_blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  
team_blob_url = f"abfss://{blob_container}@{storage_account}.dfs.core.windows.net"


# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.account.key.{storage_account}.dfs.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)
