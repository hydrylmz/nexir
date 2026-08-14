use egui::{Color32, RichText, Ui};
use std::path::PathBuf;

/// Represents a single imported media entry in the pool.
#[derive(Debug, Clone)]
pub struct MediaEntry {
    pub path: PathBuf,
    pub name: String,
    pub kind: MediaKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MediaKind {
    Video,
    Audio,
    Image,
}

impl MediaEntry {
    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
            .to_string();
        let kind = if nexir::timeline::source::is_still_image_path(&path) {
            MediaKind::Image
        } else {
            match ext.as_str() {
                "mp4" | "mov" | "mkv" | "avi" | "webm" | "mxf" | "m4v" => MediaKind::Video,
                "mp3" | "wav" | "aac" | "flac" | "ogg" | "m4a" => MediaKind::Audio,
                _ => MediaKind::Video, // default
            }
        };
        Self { path, name, kind }
    }

    pub fn icon(&self) -> &'static str {
        match self.kind {
            MediaKind::Video => "🎬",
            MediaKind::Audio => "🔊",
            MediaKind::Image => "🖼",
        }
    }
}

/// State owned by `NexirApp` for the media pool panel.
pub struct MediaPoolState {
    pub entries: Vec<MediaEntry>,
    pub selected: Option<usize>,
    pub pending_import: Option<Vec<PathBuf>>,
    pub dragging_item: Option<MediaEntry>,
}

impl Default for MediaPoolState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            selected: None,
            pending_import: None,
            dragging_item: None,
        }
    }
}

pub fn draw(ui: &mut Ui, state: &mut MediaPoolState) {
    // ── Header ────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.strong(RichText::new("Media Pool").color(Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let import_btn = egui::Button::new(RichText::new("+ Import").color(Color32::WHITE))
                .fill(Color32::from_rgb(0, 130, 220));
            if ui.add(import_btn).clicked() {
                // Trigger native file dialog — runs on main thread synchronously via rfd blocking API
                let files = rfd::FileDialog::new()
                    .add_filter(
                        "Media",
                        &[
                            "mp4", "mov", "mkv", "avi", "webm", "mxf", "m4v", "mp3", "wav", "aac",
                            "flac", "ogg", "m4a", "png", "jpg", "jpeg", "bmp", "tiff", "webp",
                        ],
                    )
                    .set_title("Import Media")
                    .pick_files();
                if let Some(paths) = files {
                    state.pending_import = Some(paths);
                }
            }
        });
    });

    ui.separator();

    // ── Process any pending imports ────────────────────────────────────
    if let Some(paths) = state.pending_import.take() {
        for path in paths {
            let entry = MediaEntry::from_path(path);
            // Avoid duplicates
            if !state.entries.iter().any(|e| e.path == entry.path) {
                state.entries.push(entry);
            }
        }
    }

    // ── Entry list ────────────────────────────────────────────────────
    if state.entries.is_empty() {
        ui.add_space(20.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("📂").size(32.0));
            ui.add_space(6.0);
            ui.label(
                RichText::new("No media imported.\nClick + Import to add files.")
                    .color(Color32::GRAY)
                    .size(12.0),
            );
        });
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (i, entry) in state.entries.iter().enumerate() {
                let is_selected = state.selected == Some(i);
                let bg = if is_selected {
                    Color32::from_rgb(0, 100, 200)
                } else if i % 2 == 0 {
                    Color32::from_rgb(32, 32, 32)
                } else {
                    Color32::from_rgb(28, 28, 28)
                };

                let response = egui::Frame::none()
                    .fill(bg)
                    .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(entry.icon()).size(18.0));
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new(&entry.name)
                                    .color(if is_selected {
                                        Color32::WHITE
                                    } else {
                                        Color32::LIGHT_GRAY
                                    })
                                    .size(12.0),
                            );
                        });
                    })
                    .response
                    .interact(egui::Sense::click_and_drag());

                if response.clicked() {
                    state.selected = Some(i);
                }

                if response.drag_started() {
                    state.dragging_item = Some(entry.clone());
                }
                if response.dragged() {
                    egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("drag_tooltip"), |ui| {
                        ui.label(format!("{} {}", entry.icon(), entry.name));
                    });
                }
            }
        });
}
