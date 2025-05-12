output "jenkins_external_ip" {
  value       = google_compute_instance.jenkins.network_interface[0].access_config[0].nat_ip
  description = "Jenkins external IP (accessible via port 80)"
}
