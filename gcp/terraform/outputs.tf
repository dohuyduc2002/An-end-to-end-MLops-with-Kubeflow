output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "kubernetes_endpoint" {
  description = "Kubernetes API server endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "project_id" {
  description = "GCP Project ID used"
  value       = var.project_id
}
