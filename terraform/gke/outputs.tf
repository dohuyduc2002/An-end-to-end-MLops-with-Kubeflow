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

output "node_pool_1_name" {
  description = "Name of the first node pool (small)"
  value       = google_container_node_pool.nodepool_1.name
}

output "node_pool_2_name" {
  description = "Name of the second node pool (large)"
  value       = google_container_node_pool.nodepool_2.name
}

output "location" {
  description = "Zone where the GKE cluster is deployed"
  value       = var.zone
}
