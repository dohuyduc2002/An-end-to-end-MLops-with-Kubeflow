resource "google_compute_instance" "jenkins" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "${var.image_project}/${var.image_family}"
      size  = var.disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = var.network
    access_config {}
  }

  metadata_startup_script = templatefile("${path.module}/startup.sh.tpl", {
    jenkins_image = var.jenkins_image
    dns_hosts     = var.dns_hosts
  })

  tags = ["jenkins"]
}

resource "google_compute_firewall" "jenkins_allow" {
  name    = "jenkins-allow-http"
  network = var.network

  allow {
    protocol = "tcp"
    ports    = ["80", "50000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["jenkins"]
}
