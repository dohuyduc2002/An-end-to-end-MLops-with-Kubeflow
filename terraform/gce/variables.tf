variable "project_id" {
  type    = string
  default = "mlops-fsds"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-c"
}

variable "credentials_file" {
  type    = string
  default = "gcp-key.json"
}

variable "vm_name" {
  type    = string
  default = "jenkins-vm"
}

variable "machine_type" {
  type    = string
  default = "e2-standard-2"
}

variable "disk_size_gb" {
  type    = number
  default = 30
}

variable "image_family" {
  type    = string
  default = "ubuntu-2004-lts"
}

variable "image_project" {
  type    = string
  default = "ubuntu-os-cloud"
}

variable "network" {
  type    = string
  default = "default"
}

variable "jenkins_image" {
  description = "Docker image to run Jenkins"
  type        = string
  default     = "microwave1005/custom-jenkins:latest"
}

variable "dns_hosts" {
  description = "List of IP-hostname mappings to add to /etc/hosts"
  type = list(object({
    ip       = string
    hostname = string
  }))
  default = []
}
