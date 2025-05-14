output "ci_build_trigger_id" {
  value = google_cloudbuild_trigger.ci_build.id
}

output "manual_staging_trigger_id" {
  value = google_cloudbuild_trigger.manual_staging.id
}

output "manual_prod_trigger_id" {
  value = google_cloudbuild_trigger.manual_prod.id
}

output "cloud_function_url" {
  value = google_cloudfunctions_function.notify_discord.https_trigger_url
}

output "gke_cluster_endpoint" {
  value = data.google_container_cluster.gke.endpoint
}

output "gke_cluster_name" {
  value = data.google_container_cluster.gke.name
}
