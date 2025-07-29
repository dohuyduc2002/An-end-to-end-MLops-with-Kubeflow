terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "6.45.0"
    }
  }
}

provider "google" {
  project     = "mlops-465414"
  region      = "us-central1"
  credentials = file("gcp-key.json")
}