variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "zone" {
  description = "GCP Zone"
  type        = string
}

variable "credentials_file" {
  description = "Path to GCP credentials JSON file"
  type        = string
}

# ─── Shared ───────────────────────────────────────────────
variable "env_label" {
  description = "Environment label for node pool"
  type        = string
  default     = "production"
}

# ─── Node Pool 1 (Small Node) ─────────────────────────────
variable "node_1_machine_type" {
  type        = string
  default     = "e2-standard-2"
  description = "Machine type for node pool 1"
}

variable "node_1_disk_size_gb" {
  type        = number
  default     = 50
}

variable "node_1_disk_type" {
  type        = string
  default     = "pd-ssd"
}

variable "node_1_preemptible" {
  type        = bool
  default     = true
}

# ─── Node Pool 2 (Large Node) ─────────────────────────────
variable "node_2_machine_type" {
  type        = string
  default     = "e2-standard-4"
  description = "Machine type for node pool 2"
}

variable "node_2_disk_size_gb" {
  type        = number
  default     = 50
}

variable "node_2_disk_type" {
  type        = string
  default     = "pd-ssd"
}

variable "node_2_preemptible" {
  type        = bool
  default     = true
}
