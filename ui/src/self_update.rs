use semver::Version;
use std::sync::mpsc::{self, Receiver, Sender};

const REPO_OWNER: &str = "hydrylmz";
const REPO_NAME: &str = "nexir";
const BIN_NAME: &str = "nexir";
const WINDOWS_TARGET: &str = "x86_64-pc-windows-msvc";
pub const WINDOWS_ASSET: &str = "nexir-x86_64-pc-windows-msvc.zip";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateStatus {
    Checking,
    UpToDate,
    Available { version: String },
    Installing { version: String },
    Installed { version: String },
    RestartError { version: String, message: String },
    Error { message: String, can_retry: bool },
    Unsupported,
}

#[derive(Clone, Copy, Debug)]
pub struct UpdateView<'a> {
    pub status: &'a UpdateStatus,
}

enum UpdateEvent {
    CheckCompleted(Result<Option<String>, String>),
    InstallCompleted {
        version: String,
        result: Result<(), String>,
    },
}

pub struct UpdaterState {
    status: UpdateStatus,
    event_tx: Sender<UpdateEvent>,
    event_rx: Receiver<UpdateEvent>,
}

impl UpdaterState {
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::channel();
        let mut state = Self {
            status: if cfg!(windows) {
                UpdateStatus::Checking
            } else {
                UpdateStatus::Unsupported
            },
            event_tx,
            event_rx,
        };
        state.start_check();
        state
    }

    pub fn view(&self) -> UpdateView<'_> {
        UpdateView {
            status: &self.status,
        }
    }

    pub fn ready_to_restart(&self) -> bool {
        matches!(self.status, UpdateStatus::Installed { .. })
    }

    pub fn poll(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            self.apply_event(event);
        }
    }

    pub fn retry_check(&mut self) {
        if matches!(
            self.status,
            UpdateStatus::Error {
                can_retry: true,
                ..
            }
        ) {
            self.status = UpdateStatus::Checking;
            self.start_check();
        }
    }

    pub fn install(&mut self) -> bool {
        let UpdateStatus::Available { version } = &self.status else {
            return false;
        };
        let version = version.clone();
        self.status = UpdateStatus::Installing {
            version: version.clone(),
        };
        spawn_install(self.event_tx.clone(), version);
        true
    }

    pub fn dismiss_error(&mut self) {
        if matches!(self.status, UpdateStatus::Error { .. }) {
            self.status = UpdateStatus::UpToDate;
        }
    }

    pub fn restart(&self) -> Result<(), String> {
        let version = match &self.status {
            UpdateStatus::Installed { version } | UpdateStatus::RestartError { version, .. } => {
                version
            }
            _ => return Ok(()),
        };
        let executable = std::env::current_exe().map_err(|error| {
            format!("The update installed, but Nexir could not locate itself: {error}")
        })?;
        std::process::Command::new(executable)
            .spawn()
            .map_err(|error| {
                format!("The update installed, but Nexir could not restart: {error}")
            })?;
        log::info!("Restarting Nexir after installing v{version}");
        Ok(())
    }

    pub fn set_restart_error(&mut self, message: String) {
        let version = match &self.status {
            UpdateStatus::Installed { version } | UpdateStatus::RestartError { version, .. } => {
                version.clone()
            }
            _ => return,
        };
        self.status = UpdateStatus::RestartError { version, message };
    }

    fn start_check(&mut self) {
        #[cfg(windows)]
        {
            let tx = self.event_tx.clone();
            std::thread::spawn(move || {
                let result = check_latest_version().map_err(|error| error.to_string());
                let _ = tx.send(UpdateEvent::CheckCompleted(result));
            });
        }
    }

    fn apply_event(&mut self, event: UpdateEvent) {
        match event {
            UpdateEvent::CheckCompleted(Ok(Some(version))) => {
                self.status = UpdateStatus::Available { version };
            }
            UpdateEvent::CheckCompleted(Ok(None)) => self.status = UpdateStatus::UpToDate,
            UpdateEvent::CheckCompleted(Err(message)) => {
                log::warn!("Update check failed: {message}");
                self.status = UpdateStatus::Error {
                    message: format!("Update check failed: {message}"),
                    can_retry: true,
                };
            }
            UpdateEvent::InstallCompleted {
                version,
                result: Ok(()),
            } => self.status = UpdateStatus::Installed { version },
            UpdateEvent::InstallCompleted {
                result: Err(message),
                ..
            } => {
                log::error!("Update installation failed: {message}");
                self.status = UpdateStatus::Error {
                    message: format!("Update failed: {message}"),
                    can_retry: true,
                };
            }
        }
    }
}

#[cfg(windows)]
fn check_latest_version() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let releases = self_update::backends::github::ReleaseList::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .with_target(WINDOWS_ASSET)
        .build()?
        .fetch()?;
    Ok(select_update(
        env!("CARGO_PKG_VERSION"),
        releases.iter().map(|release| {
            (
                release.version.as_str(),
                release.assets.iter().map(|asset| asset.name.as_str()),
            )
        }),
    )?)
}

#[cfg(windows)]
fn spawn_install(tx: Sender<UpdateEvent>, version: String) {
    std::thread::spawn(move || {
        let result = install_version(&version).map_err(|error| error.to_string());
        let _ = tx.send(UpdateEvent::InstallCompleted { version, result });
    });
}

#[cfg(not(windows))]
fn spawn_install(tx: Sender<UpdateEvent>, version: String) {
    let _ = tx.send(UpdateEvent::InstallCompleted {
        version,
        result: Err("Automatic updates are currently available on Windows only".into()),
    });
}

#[cfg(windows)]
fn install_version(version: &str) -> Result<(), Box<dyn std::error::Error>> {
    let updater = self_update::backends::github::Update::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .bin_name(BIN_NAME)
        .target(WINDOWS_TARGET)
        .identifier(WINDOWS_ASSET)
        .target_version_tag(&format!("v{version}"))
        .current_version(env!("CARGO_PKG_VERSION"))
        .show_download_progress(false)
        .show_output(false)
        .no_confirm(true)
        .build()?;
    let status = updater.update()?;
    if !status.updated() {
        return Err(format!(
            "GitHub returned v{}, but no update was installed",
            status.version()
        )
        .into());
    }
    Ok(())
}

fn select_update<'a, I, A>(current: &str, releases: I) -> Result<Option<String>, semver::Error>
where
    I: IntoIterator<Item = (&'a str, A)>,
    A: IntoIterator<Item = &'a str>,
{
    let current = Version::parse(current)?;
    let mut newest = None;
    for (version, assets) in releases {
        if !assets.into_iter().any(|asset| asset == WINDOWS_ASSET) {
            continue;
        }
        let Ok(version) = Version::parse(version.trim_start_matches('v')) else {
            continue;
        };
        if version > current && newest.as_ref().is_none_or(|known| version > *known) {
            newest = Some(version);
        }
    }
    Ok(newest.map(|version| version.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_requires_newer_semver_and_exact_asset() {
        let selected = select_update(
            "0.1.0",
            [
                ("v0.2.0", ["other.zip"]),
                ("v0.1.1", [WINDOWS_ASSET]),
                ("v0.3.0", [WINDOWS_ASSET]),
                ("nightly", [WINDOWS_ASSET]),
            ],
        )
        .unwrap();
        assert_eq!(selected.as_deref(), Some("0.3.0"));
    }

    #[test]
    fn empty_or_old_releases_are_up_to_date() {
        let empty: [(&str, [&str; 0]); 0] = [];
        assert_eq!(select_update("1.0.0", empty).unwrap(), None);
        assert_eq!(
            select_update("1.0.0", [("v1.0.0", [WINDOWS_ASSET])]).unwrap(),
            None
        );
    }

    #[test]
    fn check_and_install_events_drive_state() {
        let (tx, rx) = mpsc::channel();
        let mut state = UpdaterState {
            status: UpdateStatus::Checking,
            event_tx: tx,
            event_rx: rx,
        };
        state.apply_event(UpdateEvent::CheckCompleted(Ok(Some("0.2.0".into()))));
        assert_eq!(
            state.status,
            UpdateStatus::Available {
                version: "0.2.0".into()
            }
        );
        state.apply_event(UpdateEvent::InstallCompleted {
            version: "0.2.0".into(),
            result: Ok(()),
        });
        assert_eq!(
            state.status,
            UpdateStatus::Installed {
                version: "0.2.0".into()
            }
        );
    }

    #[test]
    fn errors_remain_visible_and_retryable() {
        let (tx, rx) = mpsc::channel();
        let mut state = UpdaterState {
            status: UpdateStatus::Checking,
            event_tx: tx,
            event_rx: rx,
        };
        state.apply_event(UpdateEvent::CheckCompleted(Err("offline".into())));
        assert!(matches!(
            state.status,
            UpdateStatus::Error {
                can_retry: true,
                ..
            }
        ));
        state.dismiss_error();
        assert_eq!(state.status, UpdateStatus::UpToDate);
    }
}
