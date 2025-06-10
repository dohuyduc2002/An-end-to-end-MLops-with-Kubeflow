resource "azurerm_resource_group" "jenkins_rg" {
  name     = var.resource_group_name
  location = var.location
}

# setup a virtual network, with the range 10.0.0.0/16
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.vm_name}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = var.location
  resource_group_name = azurerm_resource_group.jenkins_rg.name
}

# setup a subnet, with the range 10.0.2.0/24 
resource "azurerm_subnet" "subnet" {
  name                 = "${var.vm_name}-subnet"
  resource_group_name  = azurerm_resource_group.jenkins_rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.2.0/24"]
}

# setup a network security group, with rules to allow inbound traffic on port 8080 (Jenkins), port 22 (SSH), and port 50000 (Jenkins agent)
resource "azurerm_network_security_group" "nsg" {
  name                = "${var.vm_name}-nsg"
  location            = var.location
  resource_group_name = azurerm_resource_group.jenkins_rg.name

  security_rule {
    name                       = "AllowJenkins"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    destination_port_range     = "8080"
    source_address_prefix      = "*"
    source_port_range          = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                   = "AllowSSH"
    priority               = 1002
    direction              = "Inbound"
    access                 = "Allow"
    protocol               = "Tcp"
    destination_port_range = "22"
    source_address_prefix  = "*"
    source_port_range      = "*"
    destination_address_prefix = "*"
  }

    security_rule {
    name                       = "AllowJenkinsAgent"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    destination_port_range     = "50000"
    source_address_prefix      = "*"
    source_port_range          = "*"
    destination_address_prefix = "*"
  }

}

# setup a public IP address, which will be used to access the VM
resource "azurerm_public_ip" "public_ip" {
  name                = "${var.vm_name}-public-ip"
  location            = var.location
  resource_group_name = azurerm_resource_group.jenkins_rg.name
  allocation_method   = "Dynamic"
  sku                 = "Basic"
}

# setup a network interface, which will be used to connect the VM to the virtual network and public IP address
resource "azurerm_network_interface" "nic" {
  name                = "${var.vm_name}-nic"
  location            = var.location
  resource_group_name = azurerm_resource_group.jenkins_rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.public_ip.id
  }
}

# associate the network interface with the network security group
resource "azurerm_network_interface_security_group_association" "nic_nsg" {
  network_interface_id      = azurerm_network_interface.nic.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

resource "azurerm_linux_virtual_machine" "jenkins_vm" {
  name                  = var.vm_name
  resource_group_name   = azurerm_resource_group.jenkins_rg.name
  location              = var.location
  size                  = "Standard_A2_v2"
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.nic.id]

  priority        = "Spot" # Use Spot VM for cost efficiency
  eviction_policy = "Deallocate" # Deallocate when evicted by Azure

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 50
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(var.public_key_path)
  }

  disable_password_authentication = true
  custom_data                     = filebase64("${path.module}/cloud-init.yaml")
}
