# --------------------------
# Pub/Sub for build events
# --------------------------
resource "google_pubsub_topic" "build_notifications" {
  name = "build-notifications"
}

resource "google_pubsub_topic_iam_member" "build_pub" {
  topic  = google_pubsub_topic.build_notifications.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
}

# --------------------------
# Secret for Discord webhook
# --------------------------
resource "google_secret_manager_secret" "discord" {
  secret_id  = "discord-webhook"
  replication { automatic = true }
}

resource "google_secret_manager_secret_version" "discord_ver" {
  secret      = google_secret_manager_secret.discord.id
  secret_data = var.discord_webhook
}

# --------------------------
# Cloud Function: notify_discord
# --------------------------
locals {
  cf_name   = "notify-discord"
  cf_path   = "${path.module}/../cloud_functions/notify_discord"
  zip_path  = "${local.cf_path}.zip"
  zip_file  = "notify_discord.zip"
}

resource "google_storage_bucket" "cf_bucket" {
  name          = "${var.project_id}-cf-bucket"
  location      = var.region
  force_destroy = true
}

data "archive_file" "cf_zip" {
  type        = "zip"
  source_dir  = local.cf_path
  output_path = local.zip_path
}

resource "google_storage_bucket_object" "cf_zip" {
  name   = local.zip_file
  bucket = google_storage_bucket.cf_bucket.name
  source = data.archive_file.cf_zip.output_path
}

resource "google_cloudfunctions_function" "notify_discord" {
  name                  = local.cf_name
  runtime               = "python39"
  entry_point           = "notify_discord"
  region                = var.region
  source_archive_bucket = google_storage_bucket.cf_bucket.name
  source_archive_object = google_storage_bucket_object.cf_zip.name
  trigger_topic         = google_pubsub_topic.build_notifications.name
  available_memory_mb   = 128

  environment_variables = {
    DISCORD_SECRET = google_secret_manager_secret.discord.name
  }
}

resource "google_cloudfunctions_function_iam_member" "secret_access" {
  project        = var.project_id
  region         = var.region
  cloud_function = google_cloudfunctions_function.notify_discord.name
  role           = "roles/secretmanager.secretAccessor"
  member         = "serviceAccount:${google_cloudfunctions_function.notify_discord.service_account_email}"
}

# --------------------------
# Cloud Build triggers
# --------------------------
resource "google_cloudbuild_trigger" "ci_build" {
  name     = "ci-build-trigger"
  filename = "cloudbuild/ci-build.yaml"
  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "main"
    }
  }
}

resource "google_cloudbuild_trigger" "manual_staging" {
  name     = "manual-staging-trigger"
  filename = "cloudbuild/promote-staging.yaml"
  github {
    owner = var.github_owner
    name  = var.github_repo
  }
  disabled = true
}

resource "google_cloudbuild_trigger" "manual_prod" {
  name     = "manual-prod-deploy-trigger"
  filename = "cloudbuild/promote-prod-deploy.yaml"
  github {
    owner = var.github_owner
    name  = var.github_repo
  }
  disabled = true
}

# --------------------------
# Reference existing GKE cluster
# --------------------------
data "google_container_cluster" "gke" {
  name     = "prediction-platform"
  location = var.zone
}
