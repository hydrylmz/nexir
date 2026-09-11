use crate::self_update::{UpdateStatus, UpdateView};
use egui::{Color32, Ui};
use nexir::project::ProjectSettings;
use nexir::timeline::rational::Rational;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopBarAction {
    Undo,
    Redo,
    NewProject,
    OpenProject,
    Save,
    SaveAs,
    Export,
    InstallUpdate,
    RetryUpdate,
    DismissUpdateError,
    RestartUpdatedApp,
}

/// Persistent state for the top bar (kept alive across frames via egui memory).
#[derive(Default, Clone)]
struct TopBarState {
    about_open: bool,
}

pub fn draw(
    ui: &mut Ui,
    can_undo: bool,
    can_redo: bool,
    settings: &mut ProjectSettings,
    updater: UpdateView<'_>,
) -> Option<TopBarAction> {
    // Load/store persistent state in egui memory so it survives across frames.
    let mut state = ui.ctx().data(|d| {
        d.get_temp::<TopBarState>(egui::Id::new("top_bar_state"))
            .unwrap_or_default()
    });
    let mut action = None;

    egui::menu::bar(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("New Project").clicked() {
                action = Some(TopBarAction::NewProject);
                ui.close_menu();
            }
            if ui.button("Open Project...").clicked() {
                action = Some(TopBarAction::OpenProject);
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Save").clicked() {
                action = Some(TopBarAction::Save);
                ui.close_menu();
            }
            if ui.button("Save As...").clicked() {
                action = Some(TopBarAction::SaveAs);
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Exit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
        ui.menu_button("Edit", |ui| {
            if ui
                .add_enabled(can_undo, egui::Button::new("Undo"))
                .clicked()
            {
                action = Some(TopBarAction::Undo);
                ui.close_menu();
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("Redo"))
                .clicked()
            {
                action = Some(TopBarAction::Redo);
                ui.close_menu();
            }
        });
        ui.menu_button("View", |ui| {
            if ui.button("Reset Layout").clicked() {
                ui.ctx().memory_mut(|mem| *mem = Default::default());
                ui.close_menu();
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("About nexir").clicked() {
                state.about_open = true;
                ui.close_menu();
            }
        });

        // ── Project settings (canvas resolution + frame rate) ────────────────
        ui.separator();

        // Resolution preset combo
        egui::ComboBox::from_id_source("proj_resolution")
            .selected_text(resolution_label(settings.width, settings.height))
            .width(110.0)
            .show_ui(ui, |ui| {
                for &(w, h) in RESOLUTION_PRESETS {
                    if ui
                        .selectable_label(
                            settings.width == w && settings.height == h,
                            resolution_label(w, h),
                        )
                        .clicked()
                    {
                        settings.width = w;
                        settings.height = h;
                    }
                }
            });

        // Frame rate combo
        egui::ComboBox::from_id_source("proj_fps")
            .selected_text(fps_label(settings.frame_rate))
            .width(70.0)
            .show_ui(ui, |ui| {
                for &(n, d) in FPS_PRESETS {
                    let r = Rational::new(n, d);
                    if ui
                        .selectable_label(
                            settings.frame_rate.num == n && settings.frame_rate.den == d,
                            fps_label(r),
                        )
                        .clicked()
                    {
                        settings.frame_rate = r;
                    }
                }
            });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // CapCut style bright blue export button
            let export_btn = egui::Button::new(egui::RichText::new("Export").color(Color32::WHITE))
                .fill(Color32::from_rgb(0, 153, 255));

            if ui.add(export_btn).clicked() {
                action = Some(TopBarAction::Export);
            }

            ui.add_space(8.0);
            ui.label("nexir Editor");

            match updater.status {
                UpdateStatus::Available { version } => {
                    ui.add_space(8.0);
                    if ui
                        .button(
                            egui::RichText::new(format!("v{version} available — Update & Restart"))
                                .color(Color32::from_rgb(120, 210, 255)),
                        )
                        .clicked()
                    {
                        action = Some(TopBarAction::InstallUpdate);
                    }
                }
                UpdateStatus::Installing { version } => {
                    ui.add_space(8.0);
                    ui.spinner();
                    ui.label(format!("Installing v{version}…"));
                }
                UpdateStatus::Installed { version } => {
                    ui.add_space(8.0);
                    if ui.button(format!("Restart to use v{version}")).clicked() {
                        action = Some(TopBarAction::RestartUpdatedApp);
                    }
                }
                UpdateStatus::RestartError { version, message } => {
                    ui.add_space(8.0);
                    ui.colored_label(Color32::from_rgb(255, 140, 120), message);
                    if ui.button(format!("Retry restart for v{version}")).clicked() {
                        action = Some(TopBarAction::RestartUpdatedApp);
                    }
                }
                UpdateStatus::Error { message, can_retry } => {
                    ui.add_space(8.0);
                    ui.colored_label(Color32::from_rgb(255, 140, 120), message);
                    if *can_retry && ui.small_button("Retry").clicked() {
                        action = Some(TopBarAction::RetryUpdate);
                    }
                    if ui.small_button("Dismiss").clicked() {
                        action = Some(TopBarAction::DismissUpdateError);
                    }
                }
                UpdateStatus::Checking | UpdateStatus::UpToDate | UpdateStatus::Unsupported => {}
            }
        });
    });

    // ── About dialog ────────────────────────────────────────────────────────
    if state.about_open {
        egui::Window::new("About nexir")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("nexir")
                            .size(28.0)
                            .strong()
                            .color(Color32::from_rgb(0, 153, 255)),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Video Editor")
                            .size(13.0)
                            .color(Color32::GRAY),
                    );
                    ui.add_space(12.0);
                    ui.label("A fast, GPU-accelerated non-linear video editor");
                    ui.label("built in Rust with wgpu + egui.");
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .monospace()
                            .color(Color32::LIGHT_GRAY),
                    );
                    ui.add_space(16.0);
                    if ui.button("  Close  ").clicked() {
                        state.about_open = false;
                    }
                    ui.add_space(8.0);
                });
            });
    }

    // Persist state back into egui memory.
    ui.ctx()
        .data_mut(|d| d.insert_temp(egui::Id::new("top_bar_state"), state));
    action
}

const RESOLUTION_PRESETS: &[(u32, u32)] = &[
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1080, 1920),
    (1080, 1080),
    (1280, 720),
    (854, 480),
];

const FPS_PRESETS: &[(i64, i64)] = &[(24, 1), (25, 1), (30, 1), (60, 1), (120, 1)];

fn resolution_label(w: u32, h: u32) -> &'static str {
    match (w, h) {
        (3840, 2160) => "3840x2160 4K",
        (2560, 1440) => "2560x1440 2K",
        (1920, 1080) => "1920x1080 FHD",
        (1080, 1920) => "1080x1920 9:16",
        (1080, 1080) => "1080x1080 1:1",
        (1280, 720) => "1280x720 HD",
        (854, 480) => "854x480 SD",
        _ => "Custom",
    }
}

fn fps_label(r: Rational) -> &'static str {
    match (r.num, r.den) {
        (24, 1) => "24 fps",
        (25, 1) => "25 fps",
        (30, 1) => "30 fps",
        (60, 1) => "60 fps",
        (120, 1) => "120 fps",
        _ => "? fps",
    }
}
