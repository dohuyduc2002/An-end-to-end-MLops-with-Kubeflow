# resource "google_sql_database_instance" "instance" {
#   name             = "ml-instance"
#   region           = "us-central1"
#   database_version = "POSTGRES_17"
#   project         = "mlops-465414"


#   settings {
#     tier = "db-g1-small"
#     edition = "ENTERPRISE"
#     disk_type = "PD_HDD"
#   }

#   deletion_protection = false
# }

# resource "google_sql_user" "user" {
#   name     = "ducdh"
#   instance = google_sql_database_instance.instance.name
#   password = "huyduc2002"
# }

# resource "google_sql_database" "database" {
#   name     = "homecredit"
#   instance = google_sql_database_instance.instance.name
#   deletion_policy = "DELETE"
# }

# # resource "null_resource" "init_sql" {
# #   depends_on = [
# #     google_sql_database.database,
# #     google_sql_user.user
# #   ]
# #   provisioner "local-exec" {
# #     command = <<EOT
# # gcloud auth activate-service-account --key-file=gcp-key.json

# # gcloud config set project mlops-465414
# # gcloud sql connect ${google_sql_database_instance.instance.name} \
# #   --user=ducdh --project=mlops-465414 --database=homecredit < init_tbl.sql
# # EOT
# #   }
# # }

# resource "google_bigquery_dataset" "homecredit_dataset" {
#   dataset_id                  = "homecredit_ds" 
#   friendly_name               = "Home Credit Dataset"
#   description                 = "Dataset for Home Credit tables"
#   location                    = "US"
#   delete_contents_on_destroy  = true
#   project                     = "mlops-465414"
# }

