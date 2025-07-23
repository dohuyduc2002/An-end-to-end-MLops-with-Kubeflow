resource "azurerm_resource_group" "aks_rg" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.aks_name
  location            = azurerm_resource_group.aks_rg.location
  resource_group_name = azurerm_resource_group.aks_rg.name
  dns_prefix          = "${var.aks_name}-dns"

  default_node_pool {
    name                = "default"
    node_count          = 1
    vm_size             = "Standard_B4as_v2"
    os_disk_size_gb     = 50
    auto_scaling_enabled = false
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    environment = "production"
  }

  network_profile {
    network_plugin = "azure"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "small" {
  name                  = "small"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_A2_v2"
  node_count            = 1
  os_disk_size_gb       = 50
  mode                  = "User"    
  orchestrator_version  = azurerm_kubernetes_cluster.aks.kubernetes_version
  node_labels           = { pool = "small" }
  auto_scaling_enabled  = false
}

