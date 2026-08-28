// ui/src/layout/recovery_dialog.rs
//
// Modal dialog rendered at startup when a stale autosave file is detected.
// Gives the user the choice to restore the unsaved changes or discard them.

use egui::{Color32, Context, RichText, Window};
use crate::autosave::RecoveryInfo;
use std::time::SystemTime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    Restore,
    Discard,
}

fn format_relative_time(mtime: SystemTime) -> String {
    if let Ok(elapsed) = mtime.elapsed() {
        let secs = elapsed.as_secs();
        if secs < 60 {
            format!("{} seconds ago", secs)
        } else if secs < 3600 {
            format!("{} minutes ago", secs / 60)
        } else if secs < 86400 {
            format!("{} hours ago", secs / 3600)
        } else {
            format!("{} days ago", secs / 86400)
        }
    } else {
        "recently".to_string()
    }
}

pub fn draw(ctx: &Context, info: &RecoveryInfo) -> Option<RecoveryAction> {
    let mut action = None;

    Window::new("Crash Recovery — Unsaved Project Found")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.set_min_width(420.0);

            ui.heading(RichText::new("⚠️ Unsaved Changes Detected").color(Color32::from_rgb(255, 200, 80)));
            ui.add_space(8.0);

            ui.label(
                "Nexir detected an autosave file from a previous session that was not closed cleanly. \
                 Would you like to recover your project?"
            );

            ui.add_space(10.0);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Project:").strong());
                    ui.label(&info.project_name);
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Autosave:").strong());
                    ui.label(info.autosave_path.to_string_lossy());
                });
                if let Some(mtime) = info.modified {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Saved:").strong());
                        ui.label(format_relative_time(mtime));
                    });
                }
            });

            ui.add_space(16.0);
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Restore Project").strong().color(Color32::from_rgb(100, 220, 100))).clicked() {
                    action = Some(RecoveryAction::Restore);
                }

                if ui.button(RichText::new("Discard Autosave").color(Color32::from_rgb(240, 100, 100))).clicked() {
                    action = Some(RecoveryAction::Discard);
                }
            });
        });

    action
}
