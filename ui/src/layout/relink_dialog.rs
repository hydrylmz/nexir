// ui/src/layout/relink_dialog.rs
//
// Dialog for detecting and relinking missing/offline media files.

use egui::{Color32, Context, RichText, Window};
use nexir::project::Project;
use nexir::timeline::ids::SourceId;
use std::path::PathBuf;

#[derive(Default)]
pub struct RelinkDialogState {
    pub open: bool,
    pub missing_sources: Vec<(SourceId, PathBuf)>,
}

impl RelinkDialogState {
    /// Refresh the list of missing media sources from the project source registry.
    pub fn refresh(&mut self, project: &Project) {
        let sources = project.sources.read().unwrap();
        self.missing_sources = sources.offline_sources();
        if !self.missing_sources.is_empty() {
            self.open = true;
        }
    }
}

pub fn draw(
    ctx: &Context,
    state: &mut RelinkDialogState,
    project: &mut Project,
    still_cache: &std::sync::Mutex<crate::image_still::StillImageCache>,
) {
    if !state.open || state.missing_sources.is_empty() {
        state.open = false;
        return;
    }

    let mut relink_requests: Vec<(SourceId, PathBuf)> = Vec::new();
    let mut close_dialog = false;

    Window::new("Missing Media Relinker")
        .collapsible(false)
        .resizable(true)
        .default_width(550.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.heading(RichText::new("⚠️ Offline Media Detected").color(Color32::from_rgb(255, 180, 50)));
            ui.add_space(6.0);
            ui.label(
                "The following media files referenced by this project could not be found at their saved locations. \
                 Please relocate or relink them to restore preview and export."
            );
            ui.add_space(10.0);

            egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                for (id, path) in &state.missing_sources {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let filename = path.file_name()
                                .map(|n| n.to_string_lossy())
                                .unwrap_or_else(|| "Unknown file".into());

                            ui.vertical(|ui| {
                                ui.label(RichText::new(filename.to_string()).strong());
                                ui.label(RichText::new(path.to_string_lossy()).weak().small());
                            });

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Locate...").clicked() {
                                    let mut dialog = rfd::FileDialog::new().set_title("Relink Media File");
                                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                                        dialog = dialog.add_filter("Matching extension", &[ext]);
                                    }
                                    if let Some(parent) = path.parent() {
                                        dialog = dialog.set_directory(parent);
                                    }
                                    if let Some(new_path) = dialog.pick_file() {
                                        relink_requests.push((*id, new_path));
                                    }
                                }
                            });
                        });
                    });
                    ui.add_space(4.0);
                }
            });

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("Continue Offline / Close").clicked() {
                    close_dialog = true;
                }
            });
        });

    if close_dialog {
        state.open = false;
    }

    // Apply any successful relinks
    if !relink_requests.is_empty() {
        {
            let mut sources = project.sources.write().unwrap();
            for (id, new_path) in relink_requests {
                let _ = sources.relink(id, new_path.clone());
                still_cache.lock().unwrap().evict(&new_path);
            }
        }
        // Re-check remaining offline sources
        state.refresh(project);
    }
}
