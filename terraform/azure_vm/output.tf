output "jenkins_vm_public_ip" {
  description = "Public IP address of the Jenkins VM"
  value       = azurerm_public_ip.public_ip.ip_address
}

output "jenkins_vm_name" {
  value = azurerm_linux_virtual_machine.jenkins_vm.name
}
