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
            .into_owned();
        let ext = path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        let kind = match ext.as_str() {
            "mp4" | "mov" | "mkv" | "avi" | "webm" | "mxf" | "m4v" => MediaKind::Video,
            "mp3" | "wav" | "aac" | "flac" | "ogg" | "m4a" => MediaKind::Audio,
            "png" | "jpg" | "jpeg" | "bmp" | "webp" | "tiff" => MediaKind::Image,
            _ => MediaKind::Video,
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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LibraryPanel {
    MediaPool,
    Text,
    Effects,
}

/// State for the media pool panel.
pub struct MediaPoolState {
    pub entries: Vec<MediaEntry>,
    pub selected: Option<usize>,
    pub pending_import: Option<Vec<PathBuf>>,
    pub dragging_item: Option<MediaEntry>,
    pub active_panel: LibraryPanel,
}

impl Default for MediaPoolState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            selected: None,
            pending_import: None,
            dragging_item: None,
            active_panel: LibraryPanel::MediaPool,
        }
    }
}

impl MediaPoolState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_pending_import(&mut self) -> usize {
        let Some(paths) = self.pending_import.take() else {
            return 0;
        };
        let before = self.entries.len();
        for path in paths {
            let entry = MediaEntry::from_path(path);
            if !self.entries.iter().any(|e| e.path == entry.path) {
                self.entries.push(entry);
            }
        }
        self.entries.len() - before
    }
}

pub fn draw(ui: &mut Ui, state: &mut MediaPoolState) {
    // Flush any paths queued by the import dialog into the entries list.
    // This is called unconditionally every frame so imports are visible
    // on the very next repaint after the dialog closes.
    state.apply_pending_import();

    // Capture the full available height before splitting into columns so the
    // content area and its scroll views fill the entire panel height.
    let available_height = ui.available_height();
    ui.horizontal(|ui| {
        // --- Left Vertical Bar ---
        ui.vertical(|ui| {
            ui.set_width(40.0);
            ui.set_min_height(available_height);
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
            ui.set_min_height(available_height);
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
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter(
                        "Media Files",
                        &["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "aac", "flac", "ogg", "png", "jpg", "jpeg", "bmp", "webp"],
                    )
                    .pick_files()
                {
                    state.pending_import = Some(paths);
                }
            }
        });
    });
    ui.separator();
    ui.add_space(4.0);

    // ── Media List / Grid ─────────────────────────────────────────────
    if state.entries.is_empty() {
        ui.vertical_centered(|ui| {
            ui.add_space(30.0);
            ui.label(RichText::new("📁").size(32.0));
            ui.add_space(4.0);
            ui.label(
                RichText::new("No media imported yet")
                    .color(Color32::GRAY)
                    .italics(),
            );
            ui.label(
                RichText::new("Click '+ Import' above to add video, audio, or images.")
                    .color(Color32::DARK_GRAY)
                    .small(),
            );
        });
    } else {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (i, entry) in state.entries.iter().enumerate() {
                    let is_selected = state.selected == Some(i);
                    let bg = if is_selected {
                        Color32::from_rgb(45, 60, 80)
                    } else if i % 2 == 0 {
                        Color32::from_rgb(32, 32, 32)
                    } else {
                        Color32::from_rgb(28, 28, 28)
                    };

                    let response = egui::Frame::none()
                        .fill(bg)
                        .inner_margin(egui::Margin::symmetric(6.0, 4.0))
                        .rounding(4.0)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.horizontal(|ui| {
                                ui.label(entry.icon());
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(&entry.name)
                                        .color(if is_selected {
                                            Color32::WHITE
                                        } else {
                                            Color32::LIGHT_GRAY
                                        }),
                                );
                            });
                        })
                        .response
                        .interact(egui::Sense::click_and_drag());

                    if response.clicked() {
                        state.selected = Some(i);
                    }

                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }

                    if response.drag_started() {
                        state.dragging_item = Some(entry.clone());
                    }

                    if response.dragged() {
                        egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("drag_tooltip"), |ui| {
                            ui.horizontal(|ui| {
                                ui.label(entry.icon());
                                ui.label(RichText::new(&entry.name).strong());
                            });
                        });
                    }

                    ui.add_space(2.0);
                }
            });
    }
}
