output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "kubernetes_endpoint" {
  description = "Kubernetes API server endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "bucket_url" {
  value       = "gs://${google_storage_bucket.gcs_bucket.name}/"
  description = "The GCS bucket URL"
}


# output "cloud_sql_instance_name" {
#   description = "Name of the Cloud SQL instance"
#   value       = google_sql_database_instance.instance.name
# }

# output "bigquery_dataset_id" {
#   description = "ID of the BigQuery dataset"
#   value       = google_bigquery_dataset.homecredit_dataset.dataset_id
# }