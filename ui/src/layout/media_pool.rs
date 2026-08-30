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
    Text,
    Effect,
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
            MediaKind::Text => "T",
            MediaKind::Effect => "✨",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum LibraryPanel {
    #[default]
    MediaPool,
    Text,
    Effects,
}


/// State owned by `NexirApp` for the media pool panel.
#[derive(Default)]
pub struct MediaPoolState {
    pub entries: Vec<MediaEntry>,
    pub selected: Option<usize>,
    pub pending_import: Option<Vec<PathBuf>>,
    pub dragging_item: Option<MediaEntry>,
    pub active_panel: LibraryPanel,
}

impl MediaPoolState {
    /// Take whatever the file dialog selected and fold it into the entry list.
    ///
    /// P2.7 — this is split out of [`draw`] on purpose. The only untestable part of
    /// media import is `rfd::FileDialog::pick_files`, which blocks on a human and
    /// panics without a display; everything that decides what an import MEANS —
    /// classification, de-duplication, ordering — is on this side of that seam and
    /// needs no window. `draw` calls this immediately after setting
    /// `pending_import`, so the tested path is the shipped path rather than a
    /// parallel implementation.
    ///
    /// Returns how many entries were added, which is what a caller needs to know
    /// whether anything actually happened.
    pub fn apply_pending_import(&mut self) -> usize {
        let Some(paths) = self.pending_import.take() else {
            return 0;
        };
        let before = self.entries.len();
        for path in paths {
            let entry = MediaEntry::from_path(path);
            // Re-importing a file the pool already holds must not duplicate it: the
            // list is a library keyed by path, and a second entry would give the
            // same media two selectable rows and two drag sources.
            if !self.entries.iter().any(|e| e.path == entry.path) {
                self.entries.push(entry);
            }
        }
        self.entries.len() - before
    }
}


pub fn draw(ui: &mut Ui, state: &mut MediaPoolState) {
    ui.horizontal(|ui| {
        // --- Left Vertical Bar ---
        ui.vertical(|ui| {
            ui.set_width(40.0);
            ui.add_space(8.0);
            
            let button_size = egui::vec2(36.0, 36.0);
            
            ui.vertical_centered(|ui| {
                let media_selected = state.active_panel == LibraryPanel::MediaPool;
                let media_color = if media_selected { Color32::WHITE } else { Color32::GRAY };
                let media_bg = if media_selected { Color32::from_rgb(60, 60, 60) } else { Color32::TRANSPARENT };
                let media_btn = egui::Button::new(RichText::new("🎬").color(media_color).size(20.0))
                    .fill(media_bg)
                    .min_size(button_size);
                
                if ui.add(media_btn).on_hover_text("Media Pool").clicked() {
                    state.active_panel = LibraryPanel::MediaPool;
                }
                
                ui.add_space(8.0);
                
                let text_selected = state.active_panel == LibraryPanel::Text;
                let text_color = if text_selected { Color32::WHITE } else { Color32::GRAY };
                let text_bg = if text_selected { Color32::from_rgb(60, 60, 60) } else { Color32::TRANSPARENT };
                let text_btn = egui::Button::new(RichText::new("T").color(text_color).size(20.0))
                    .fill(text_bg)
                    .min_size(button_size);
                
                if ui.add(text_btn).on_hover_text("Text Overlays").clicked() {
                    state.active_panel = LibraryPanel::Text;
                }

                ui.add_space(8.0);

                let effects_selected = state.active_panel == LibraryPanel::Effects;
                let effects_color = if effects_selected { Color32::WHITE } else { Color32::GRAY };
                let effects_bg = if effects_selected { Color32::from_rgb(60, 60, 60) } else { Color32::TRANSPARENT };
                let effects_btn = egui::Button::new(RichText::new("✨").color(effects_color).size(20.0))
                    .fill(effects_bg)
                    .min_size(button_size);

                if ui.add(effects_btn).on_hover_text("Effects").clicked() {
                    state.active_panel = LibraryPanel::Effects;
                }
            });
        });
        
        ui.separator();
        
        // --- Main Content Area ---
        ui.vertical(|ui| {
            ui.set_min_width(ui.available_width());
            match state.active_panel {
                LibraryPanel::MediaPool => draw_media_pool(ui, state),
                LibraryPanel::Text => draw_text_panel(ui, state),
                LibraryPanel::Effects => draw_effects_panel(ui, state),
            }
        });
    });
}

fn draw_effects_panel(ui: &mut Ui, state: &mut MediaPoolState) {
    ui.horizontal(|ui| {
        ui.strong(RichText::new("Effects").color(Color32::WHITE));
    });
    ui.separator();
    ui.add_space(6.0);

    let effects_list = [
        (
            "Color & Light",
            "🎨",
            "Adjust brightness, contrast, saturation, and hue",
            "nexir://internal/effect/color",
        ),
        (
            "Gaussian Blur",
            "💧",
            "Smooth gaussian blur filter",
            "nexir://internal/effect/blur",
        ),
        (
            "Sharpen",
            "⚡",
            "High-pass edge sharpening",
            "nexir://internal/effect/sharpen",
        ),
        (
            "Vignette",
            "🌑",
            "Cinematic lens vignette shading",
            "nexir://internal/effect/vignette",
        ),
        (
            "Chroma Key",
            "🟢",
            "Green & blue screen background removal",
            "nexir://internal/effect/chroma_key",
        ),
    ];

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (i, (title, icon, desc, uri)) in effects_list.iter().enumerate() {
                let effect_entry = MediaEntry {
                    path: PathBuf::from(uri),
                    name: title.to_string(),
                    kind: MediaKind::Effect,
                };

                let bg = if i % 2 == 0 {
                    Color32::from_rgb(32, 32, 32)
                } else {
                    Color32::from_rgb(28, 28, 28)
                };

                let response = egui::Frame::none()
                    .fill(bg)
                    .inner_margin(egui::Margin::symmetric(10.0, 8.0))
                    .rounding(6.0)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(*icon).size(22.0));
                            ui.add_space(8.0);
                            ui.vertical(|ui| {
                                ui.label(RichText::new(*title).size(13.0).strong().color(Color32::WHITE));
                                ui.label(RichText::new(*desc).size(11.0).color(Color32::GRAY));
                            });
                        });
                    })
                    .response
                    .interact(egui::Sense::click_and_drag());

                if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }

                if response.drag_started() {
                    state.dragging_item = Some(effect_entry.clone());
                }

                if response.dragged() {
                    egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("drag_effect_tooltip"), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(*icon).size(16.0));
                            ui.label(RichText::new(format!("{} (Drag onto a clip)", title)).strong());
                        });
                    });
                }

                ui.add_space(4.0);
            }
        });
}

fn draw_text_panel(ui: &mut Ui, state: &mut MediaPoolState) {
    ui.horizontal(|ui| {
        ui.strong(RichText::new("Text").color(Color32::WHITE));
    });
    ui.separator();
    ui.add_space(6.0);
    
    let text_entry = MediaEntry {
        path: PathBuf::from("nexir://internal/text"),
        name: "Basic Text".to_string(),
        kind: MediaKind::Text,
    };
    
    let response = egui::Frame::none()
        .fill(Color32::from_rgb(32, 32, 32))
        .inner_margin(egui::Margin::symmetric(12.0, 10.0))
        .rounding(6.0)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new("T").size(22.0).strong());
                ui.add_space(8.0);
                ui.vertical(|ui| {
                    ui.label(RichText::new("Basic Text").size(13.0).strong().color(Color32::WHITE));
                    ui.label(RichText::new("Drag onto timeline to add text overlay").size(11.0).color(Color32::GRAY));
                });
            });
        })
        .response
        .interact(egui::Sense::click_and_drag());
        
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    if response.drag_started() {
        state.dragging_item = Some(text_entry.clone());
    }
    
    if response.dragged() {
        egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("drag_text_tooltip"), |ui| {
            ui.label("T Basic Text");
        });
    }
}

fn draw_media_pool(ui: &mut Ui, state: &mut MediaPoolState) {
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
    // The classification and de-duplication live on `MediaPoolState` so they can
    // be tested without a file dialog; see `apply_pending_import`.
    state.apply_pending_import();

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
