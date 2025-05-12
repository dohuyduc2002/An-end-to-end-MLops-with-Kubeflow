resource "google_container_cluster" "primary" {
  name                     = "prediction-platform"
  location                 = var.zone
  deletion_protection      = false
  remove_default_node_pool = true
  initial_node_count       = 1

  networking_mode          = "VPC_NATIVE"
  ip_allocation_policy     {}
}

resource "google_container_node_pool" "nodepool_1" {
  name       = "nodepool-small"
  cluster    = google_container_cluster.primary.name
  location   = var.zone
  node_count = 1

  node_config {
    machine_type = var.node_1_machine_type
    disk_size_gb = var.node_1_disk_size_gb
    disk_type    = var.node_1_disk_type
    preemptible  = var.node_1_preemptible

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      env = var.env_label
      type = "small"
    }
  }
}

resource "google_container_node_pool" "nodepool_2" {
  name       = "nodepool-large"
  cluster    = google_container_cluster.primary.name
  location   = var.zone
  node_count = 1

  node_config {
    machine_type = var.node_2_machine_type
    disk_size_gb = var.node_2_disk_size_gb
    disk_type    = var.node_2_disk_type
    preemptible  = var.node_2_preemptible

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      env = var.env_label
      type = "large"
    }
  }
}
