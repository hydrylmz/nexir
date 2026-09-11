use log::{info, warn};
use std::env;

pub fn check_for_updates() {
    let repo_owner = env::var("NEXIR_REPO_OWNER").unwrap_or_else(|_| "bmiko".to_string());
    let repo_name = env::var("NEXIR_REPO_NAME").unwrap_or_else(|_| "nexir".to_string());
    let bin_name = env::var("NEXIR_BIN_NAME").unwrap_or_else(|_| "nexir".to_string());
    let current_version = env!("CARGO_PKG_VERSION");

    info!(
        "Checking for updates for {}/{} (binary: {}) at v{}",
        repo_owner, repo_name, bin_name, current_version
    );

    let result = self_update::backends::github::Update::configure()
        .repo_owner(repo_owner.as_str())
        .repo_name(repo_name.as_str())
        .bin_name(bin_name.as_str())
        .show_download_progress(true)
        .current_version(current_version)
        .no_confirm(true)
        .build()
        .map_err(|err| err.to_string())
        .and_then(|updater| updater.update().map_err(|err| err.to_string()));

    match result {
        Ok(_) => info!("Nexir is up to date or was updated successfully."),
        Err(err) => warn!("Self-update check failed: {}", err),
    }
}
