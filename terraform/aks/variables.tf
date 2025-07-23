variable "subscription_id" {
  description = "Azure Subscription ID"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "aks-rg"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "northcentralus"
}

variable "aks_name" {
  description = "AKS cluster name"
  type        = string
  default     = "dataeng-aks"
}

variable "node_count" {
  description = "Number of AKS nodes"
  type        = number
  default     = 1
}