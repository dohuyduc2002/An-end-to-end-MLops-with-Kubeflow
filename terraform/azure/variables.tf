variable "resource_group_name" {
  default = "jenkins-rg"
}

variable "location" {
  default = "Central US"
}

variable "vm_name" {
  default = "jenkins-vm"
}

variable "admin_username" {
  default = "ducdh"
}

variable "public_key_path" {
  default = "/Users/microwave/.ssh/id_rsa.pub"
}

variable "subscription_id" {
  description = "Azure Subscription ID"
  type        = string
  
}