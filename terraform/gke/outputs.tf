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

output "bucket_http_url" {
  value       = "https://storage.googleapis.com/${google_storage_bucket.gcs_bucket.name}/"
  description = "The GCS bucket HTTP URL"
}
