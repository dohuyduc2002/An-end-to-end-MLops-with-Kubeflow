resource "google_container_cluster" "primary" {
  name     = "prediction-platform"
  location = var.zone
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1

  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  cluster    = google_container_cluster.primary.name
  location   = var.zone
  node_count = 1 

  node_config {
    machine_type = "e2-standard-8"  
    disk_size_gb = 50              
    disk_type    = "pd-ssd"
    preemptible  = true
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    labels = {
      env = "production"
    }
  }
}