use egui::{Ui, RichText, Color32};
use nexir::project::Project;

pub struct InspectorState {
    // Local edit copies — written back to the project on change
    pub scale:    f32,
    pub pos_x:    f32,
    pub pos_y:    f32,
    pub rotation: f32,
    pub opacity:  f32,
    pub speed:    f32,
    pub pitch:    f32,

    /// The clip store-index we last loaded values from.
    /// Used to detect when the selection changes so we can reload.
    last_loaded_clip: Option<usize>,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            scale:    1.0,
            pos_x:    0.0,
            pos_y:    0.0,
            rotation: 0.0,
            opacity:  100.0,
            speed:    1.0,
            pitch:    0.0,
            last_loaded_clip: None,
        }
    }
}

pub fn draw(ui: &mut Ui, state: &mut InspectorState, project: &mut Project, selected_clip: Option<usize>) {
    ui.heading(RichText::new("Inspector").color(Color32::WHITE));
    ui.separator();
    ui.add_space(4.0);

    // Wrap everything in a scroll area so content is always reachable
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
        draw_inner(ui, state, project, selected_clip);
    });
}

fn draw_inner(ui: &mut Ui, state: &mut InspectorState, project: &mut Project, selected_clip: Option<usize>) {

    // ── Sync local state when selection changes ──────────────────────────
    if selected_clip != state.last_loaded_clip || selected_clip.is_some() {
        if let Some(idx) = selected_clip {
            let t = project.clips.transform_at(idx);
            state.pos_x    = t.position[0];
            state.pos_y    = t.position[1];
            state.scale    = t.scale[0];          // uniform scale (X)
            state.rotation = t.rotation.to_degrees();
            state.opacity  = project.clips.opacity_at(idx) * 100.0;
            state.speed    = project.clips.speed_at(idx);
            state.pitch    = project.clips.pitch_at(idx);
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
            ui.collapsing("⚡  Speed & Pitch", |ui| {
                ui.label(
                    RichText::new("Select a clip to edit speed and pitch.")
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
                sources.path(source_id)
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

            // ── Transform ────────────────────────────────────────────────
            let mut transform_changed = false;
            let mut opacity_changed = false;
            ui.collapsing("🎬  Transform", |ui| {
                egui::Grid::new("transform_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Scale");
                        transform_changed |= ui.add(egui::Slider::new(&mut state.scale, 0.1..=5.0).suffix("x")).changed();
                        ui.end_row();

                        ui.label("Position X");
                        transform_changed |= ui.add(egui::Slider::new(&mut state.pos_x, -1920.0..=1920.0).suffix("px")).changed();
                        ui.end_row();

                        ui.label("Position Y");
                        transform_changed |= ui.add(egui::Slider::new(&mut state.pos_y, -1080.0..=1080.0).suffix("px")).changed();
                        ui.end_row();

                        ui.label("Rotation");
                        transform_changed |= ui.add(egui::Slider::new(&mut state.rotation, -180.0..=180.0).suffix("°")).changed();
                        ui.end_row();

                        ui.label("Opacity");
                        opacity_changed |= ui.add(egui::Slider::new(&mut state.opacity, 0.0..=100.0).suffix("%")).changed();
                        ui.end_row();
                    });

                ui.add_space(4.0);
                if ui.button("↺  Reset").clicked() {
                    state.scale    = 1.0;
                    state.pos_x    = 0.0;
                    state.pos_y    = 0.0;
                    state.rotation = 0.0;
                    state.opacity  = 100.0;
                    transform_changed = true;
                    opacity_changed = true;
                }
            });
            // Write edited values back into the clip store
            if transform_changed {
                let t = project.clips.transform_at(idx);
                let mut new_t = *t;
                new_t.position = [state.pos_x, state.pos_y];
                new_t.scale    = [state.scale, state.scale];
                new_t.rotation = state.rotation.to_radians();
                project.clips.set_transform_at(idx, new_t);
            }
            if opacity_changed {
                project.clips.set_opacity_at(idx, state.opacity / 100.0);
            }

            ui.add_space(6.0);

            // ── Audio ─────────────────────────────────────────────────────
            ui.collapsing("🔊  Audio", |ui| {
                ui.label("Audio properties (coming soon).");
            });

            ui.add_space(6.0);

            // ── Speed & Pitch ──────────────────────────────────────────────
            let speed_resp = egui::CollapsingHeader::new("⚡  Speed & Pitch")
                .default_open(true)
                .id_source("speed_pitch_header")
                .show(ui, |ui| {
                    egui::Grid::new("speed_pitch_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Speed");
                            if ui.add(
                                egui::Slider::new(&mut state.speed, 0.1..=8.0)
                                    .suffix("x")
                                    .logarithmic(true),
                            ).changed() {
                                project.clips.set_speed_at(idx, state.speed);
                            }
                            ui.end_row();

                            ui.label("Pitch");
                            if ui.add(
                                egui::Slider::new(&mut state.pitch, -12.0..=12.0)
                                    .suffix(" st")
                                    .fixed_decimals(1),
                            ).changed() {
                                project.clips.set_pitch_at(idx, state.pitch);
                            }
                            ui.end_row();
                        });

                    ui.add_space(4.0);
                    // Speed presets
                    ui.label(RichText::new("Speed presets:").color(Color32::from_rgb(160, 160, 160)).small());
                    ui.horizontal(|ui| {
                        for (i, &preset) in [0.25_f32, 0.5, 1.0, 1.5, 2.0, 4.0].iter().enumerate() {
                            let btn = egui::Button::new(format!("{}×", preset)).small();
                            if ui.add(btn).clicked() {
                                state.speed = preset;
                                project.clips.set_speed_at(idx, preset);
                            }
                            let _ = i; // suppress warning
                        }
                    });
                    ui.add_space(4.0);
                    // Pitch presets
                    ui.label(RichText::new("Pitch presets:").color(Color32::from_rgb(160, 160, 160)).small());
                    ui.horizontal(|ui| {
                        for &semitones in &[-12_i32, -7, -5, 0, 5, 7, 12] {
                            let label = if semitones == 0 { "0".to_string() } else { format!("{:+}", semitones) };
                            if ui.add(egui::Button::new(format!("{} st", label)).small()).clicked() {
                                state.pitch = semitones as f32;
                                project.clips.set_pitch_at(idx, semitones as f32);
                            }
                        }
                    });
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
