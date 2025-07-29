resource "google_container_cluster" "primary" {
  name     = "kubeflow-platform"
  location = "us-central1-c"
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1

  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  cluster    = google_container_cluster.primary.name
  location   = "us-central1-c"
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
      env = "fsds"
    }
  }
}

resource "google_storage_bucket" "gcs_bucket" {
  name          = "ducdh-bucket"
  location      = "US-CENTRAL1"
  force_destroy = true
  project = "mlops-465414"
  storage_class = "STANDARD"
  requester_pays = true
  public_access_prevention = "inherited"
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_binding" "binding" {
  bucket = google_storage_bucket.gcs_bucket.name
  role = "roles/storage.admin"
  members = [
    "user:dhduc.storage1@gmail.com",
  ]
}