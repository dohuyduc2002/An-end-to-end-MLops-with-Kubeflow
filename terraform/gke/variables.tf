variable "project_id" {
  description = "GCP Project ID"
  default = "mlops-465414"
}

variable "region" {
  description = "GCP Region"
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  default     = "us-central1-c"
}

variable "credentials_file" {
  description = "Path to GCP credentials JSON file"
  default     = "gcp-key.json"
}