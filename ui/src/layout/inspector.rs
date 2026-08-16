use crate::history::HistoryState;
use egui::{Color32, RichText, Ui};
use nexir::project::Project;

pub struct InspectorState {
    // Local edit copies — written back to the project on change
    pub scale: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub rotation: f32,
    pub opacity: f32,
    pub volume: f32,
    pub pan: f32,
    pub audio_muted: bool,
    pub speed: f32,

    // Text clip editing
    pub text_content: String,
    pub font_size: f32,
    pub text_color: [f32; 4],

    /// The clip store-index we last loaded values from.
    /// Used to detect when the selection changes so we can reload.
    last_loaded_clip: Option<usize>,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pos_x: 0.0,
            pos_y: 0.0,
            rotation: 0.0,
            opacity: 100.0,
            volume: 100.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            text_content: String::new(),
            font_size: 48.0,
            text_color: [1.0, 1.0, 1.0, 1.0],
            last_loaded_clip: None,
        }
    }
}

pub fn draw(
    ui: &mut Ui,
    state: &mut InspectorState,
    project: &mut Project,
    selected_clip: Option<usize>,
    history: &mut HistoryState,
) {
    ui.heading(RichText::new("Inspector").color(Color32::WHITE));
    ui.separator();
    ui.add_space(4.0);

    // Wrap everything in a scroll area so content is always reachable
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            draw_inner(ui, state, project, selected_clip, history);
        });
}

fn draw_inner(
    ui: &mut Ui,
    state: &mut InspectorState,
    project: &mut Project,
    selected_clip: Option<usize>,
    history: &mut HistoryState,
) {
    // ── Sync local state when selection changes ──────────────────────────
    if selected_clip != state.last_loaded_clip || selected_clip.is_some() {
        if let Some(idx) = selected_clip {
            let t = project.clips.transform_at(idx);
            state.pos_x = t.position[0];
            state.pos_y = t.position[1];
            state.scale = t.scale[0]; // uniform scale (X)
            state.rotation = t.rotation.to_degrees();
            state.opacity = project.clips.opacity_at(idx) * 100.0;
            state.volume = project.clips.volume_at(idx) * 100.0;
            state.pan = project.clips.pan_at(idx) * 100.0;
            state.audio_muted = project.clips.audio_muted_at(idx);
            state.speed = project.clips.speed_at(idx);
            // Load text clip properties
            match project.clips.kind_at(idx) {
                nexir::timeline::store::ClipKind::Text { text, font_size, color } => {
                    state.text_content = text.clone();
                    state.font_size = *font_size;
                    state.text_color = *color;
                }
                _ => {
                    state.text_content = String::new();
                    state.font_size = 48.0;
                    state.text_color = [1.0, 1.0, 1.0, 1.0];
                }
            }
        } else {
            *state = InspectorState::default();
        }
        state.last_loaded_clip = selected_clip;
    }

    match selected_clip {
        None => {
            // ── Nothing selected — show placeholders like CapCut ─────────
            ui.add_space(8.0);
            ui.label(
                RichText::new("No clip selected")
                    .color(Color32::from_rgb(120, 120, 120))
                    .italics(),
            );
            ui.add_space(6.0);

            ui.collapsing("🎬  Transform", |ui| {
                ui.label(
                    RichText::new("Select a clip to edit transform.")
                        .color(Color32::from_rgb(100, 100, 100))
                        .italics(),
                );
            });
            ui.add_space(6.0);
            ui.collapsing("🔊  Audio", |ui| {
                ui.label(
                    RichText::new("Select a clip to edit audio properties.")
                        .color(Color32::from_rgb(100, 100, 100))
                        .italics(),
                );
            });
            ui.add_space(6.0);
            ui.collapsing("⚡  Speed", |ui| {
                ui.label(
                    RichText::new("Select a clip to edit speed.")
                        .color(Color32::from_rgb(100, 100, 100))
                        .italics(),
                );
            });
            ui.add_space(6.0);
            ui.collapsing("✨  Effects", |ui| {
                ui.label(
                    RichText::new("No effects applied.")
                        .color(Color32::from_rgb(100, 100, 100))
                        .italics(),
                );
            });
        }

        Some(idx) => {
            // ── Clip selected — show real editable properties ────────────
            let source_id = project.clips.source_id_at(idx);
            let clip_name = {
                let sources = project.sources.read().unwrap();
                sources
                    .path(source_id)
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| format!("Clip {}", idx))
            };

            // Clip header chip
            ui.horizontal(|ui| {
                ui.painter().rect_filled(
                    ui.available_rect_before_wrap(),
                    4.0,
                    Color32::from_rgb(30, 40, 55),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("✂  {}", clip_name))
                        .color(Color32::from_rgb(0, 200, 255))
                        .strong(),
                );
            });
            ui.add_space(6.0);

            // ── Text (only shown for Text clips) ─────────────────────────
            let is_text_clip = matches!(
                project.clips.kind_at(idx),
                nexir::timeline::store::ClipKind::Text { .. }
            );
            if is_text_clip {
                let mut text_changed = false;
                ui.collapsing("T  Text", |ui| {
                    egui::Grid::new("text_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Content");
                            text_changed |= ui.text_edit_singleline(&mut state.text_content).changed();
                            ui.end_row();

                            ui.label("Font Size");
                            text_changed |= ui
                                .add(egui::Slider::new(&mut state.font_size, 10.0..=200.0).suffix("pt"))
                                .changed();
                            ui.end_row();

                            ui.label("Color");
                            let mut rgb = [state.text_color[0], state.text_color[1], state.text_color[2]];
                            if ui.color_edit_button_rgb(&mut rgb).changed() {
                                state.text_color = [rgb[0], rgb[1], rgb[2], state.text_color[3]];
                                text_changed = true;
                            }
                            ui.end_row();
                        });
                });
                if text_changed {
                    history.record(project);
                    project.clips.set_kind_at(idx, nexir::timeline::store::ClipKind::Text {
                        text: state.text_content.clone(),
                        font_size: state.font_size,
                        color: state.text_color,
                    });
                }
                ui.add_space(6.0);
            }

            // ── Transform ────────────────────────────────────────────────
            let mut transform_changed = false;
            let mut opacity_changed = false;
            ui.collapsing("🎬  Transform", |ui| {
                egui::Grid::new("transform_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Scale");
                        transform_changed |= ui
                            .add(egui::Slider::new(&mut state.scale, 0.1..=5.0).suffix("x"))
                            .changed();
                        ui.end_row();

                        ui.label("Position X");
                        transform_changed |= ui
                            .add(egui::Slider::new(&mut state.pos_x, -(project.settings.width as f32)..=(project.settings.width as f32)).suffix("px"))
                            .changed();
                        ui.end_row();

                        ui.label("Position Y");
                        transform_changed |= ui
                            .add(egui::Slider::new(&mut state.pos_y, -(project.settings.height as f32)..=(project.settings.height as f32)).suffix("px"))
                            .changed();
                        ui.end_row();

                        ui.label("Rotation");
                        transform_changed |= ui
                            .add(egui::Slider::new(&mut state.rotation, -180.0..=180.0).suffix("°"))
                            .changed();
                        ui.end_row();

                        ui.label("Opacity");
                        opacity_changed |= ui
                            .add(egui::Slider::new(&mut state.opacity, 0.0..=100.0).suffix("%"))
                            .changed();
                        ui.end_row();
                    });

                ui.add_space(4.0);
                if ui.button("↺  Reset").clicked() {
                    state.scale = 1.0;
                    state.pos_x = 0.0;
                    state.pos_y = 0.0;
                    state.rotation = 0.0;
                    state.opacity = 100.0;
                    transform_changed = true;
                    opacity_changed = true;
                }
            });
            // Write edited values back into the clip store
            if transform_changed || opacity_changed {
                history.record(project);
            }
            if transform_changed {
                let t = project.clips.transform_at(idx);
                let mut new_t = *t;
                new_t.position = [state.pos_x, state.pos_y];
                new_t.scale = [state.scale, state.scale];
                new_t.rotation = state.rotation.to_radians();
                project.clips.set_transform_at(idx, new_t);
            }
            if opacity_changed {
                project.clips.set_opacity_at(idx, state.opacity / 100.0);
            }

            ui.add_space(6.0);

            // ── Audio ─────────────────────────────────────────────────────
            let mut audio_changed = false;
            ui.collapsing("🔊  Audio", |ui| {
                egui::Grid::new("audio_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Mute");
                        audio_changed |= ui.checkbox(&mut state.audio_muted, "").changed();
                        ui.end_row();

                        ui.label("Volume");
                        audio_changed |= ui
                            .add(
                                egui::Slider::new(&mut state.volume, 0.0..=200.0)
                                    .suffix("%")
                                    .fixed_decimals(0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Pan");
                        audio_changed |= ui
                            .add(
                                egui::Slider::new(&mut state.pan, -100.0..=100.0)
                                    .suffix("%")
                                    .fixed_decimals(0),
                            )
                            .changed();
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new("Left").small()).clicked() {
                        state.pan = -100.0;
                        audio_changed = true;
                    }
                    if ui.add(egui::Button::new("Center").small()).clicked() {
                        state.pan = 0.0;
                        audio_changed = true;
                    }
                    if ui.add(egui::Button::new("Right").small()).clicked() {
                        state.pan = 100.0;
                        audio_changed = true;
                    }
                    if ui.add(egui::Button::new("Reset").small()).clicked() {
                        state.volume = 100.0;
                        state.pan = 0.0;
                        state.audio_muted = false;
                        audio_changed = true;
                    }
                });
            });
            if audio_changed {
                history.record(project);
                project
                    .clips
                    .set_volume_at(idx, (state.volume / 100.0).clamp(0.0, 2.0));
                project
                    .clips
                    .set_pan_at(idx, (state.pan / 100.0).clamp(-1.0, 1.0));
                project.clips.set_audio_muted_at(idx, state.audio_muted);
            }

            ui.add_space(6.0);

            // ── Speed ────────────────────────────────────────────────────────
            let speed_resp = egui::CollapsingHeader::new("⚡  Speed")
                .default_open(true)
                .id_source("speed_header")
                .show(ui, |ui| {
                    egui::Grid::new("speed_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Speed");
                            if ui
                                .add(
                                    egui::Slider::new(&mut state.speed, 0.1..=8.0)
                                        .suffix("x")
                                        .logarithmic(true),
                                )
                                .changed()
                            {
                                history.record(project);
                                project.clips.set_speed_at(idx, state.speed);
                                let clip_id = project.clips.clip_id_at(idx);
                                if let Some(linked_id) =
                                    crate::layout::timeline::find_linked_clip(project, clip_id)
                                {
                                    if let Some(linked_idx) = project.clips.index_of(linked_id) {
                                        project.clips.set_speed_at(linked_idx, state.speed);
                                    }
                                }
                            }
                            ui.end_row();
                        });

                    ui.add_space(4.0);
                    // Speed presets
                    ui.label(
                        RichText::new("Speed presets:")
                            .color(Color32::from_rgb(160, 160, 160))
                            .small(),
                    );
                    ui.horizontal(|ui| {
                        for (i, &preset) in [0.25_f32, 0.5, 1.0, 1.5, 2.0, 4.0].iter().enumerate() {
                            let btn = egui::Button::new(format!("{}×", preset)).small();
                            if ui.add(btn).clicked() {
                                history.record(project);
                                state.speed = preset;
                                project.clips.set_speed_at(idx, preset);
                                let clip_id = project.clips.clip_id_at(idx);
                                if let Some(linked_id) =
                                    crate::layout::timeline::find_linked_clip(project, clip_id)
                                {
                                    if let Some(linked_idx) = project.clips.index_of(linked_id) {
                                        project.clips.set_speed_at(linked_idx, preset);
                                    }
                                }
                            }
                            let _ = i; // suppress warning
                        }
                    });
                    ui.add_space(4.0);
                });
            let _ = speed_resp;

            ui.add_space(6.0);

            // ── Effects ───────────────────────────────────────────────────
            ui.collapsing("✨  Effects", |ui| {
                ui.label("No effects applied.");
            });
        }
    }
}
