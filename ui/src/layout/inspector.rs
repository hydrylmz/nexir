use crate::history::HistoryState;
use egui::{Color32, RichText, Ui};
use nexir::project::Project;
use nexir::timeline::transform::{BlendMode, CropRect};

pub struct InspectorState {
    // Local edit copies — written back to the project on change
    pub scale: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub rotation: f32,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub crop_left: f32,
    pub crop_top: f32,
    pub crop_right: f32,
    pub crop_bottom: f32,
    pub crop_feather: f32,

    pub volume: f32,
    pub pan: f32,
    pub audio_muted: bool,
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
    pub speed: f32,

    // Text clip editing
    pub text_content: String,
    pub font_size: f32,
    pub text_color: [f32; 4],
    // Stroke
    pub stroke_enabled: bool,
    pub stroke_color: [f32; 3],
    pub stroke_width: f32,
    // Background
    pub bg_enabled: bool,
    pub bg_color: [f32; 4],
    pub bg_padding: f32,

    // Effects
    pub chroma_key_enabled: bool,
    pub chroma_key_color: [f32; 3],
    pub chroma_key_tolerance: f32,
    pub chroma_key_softness: f32,

    /// When true, the next viewport click picks a color for chroma key.
    pub eyedropper_active: bool,

    /// The clip store-index we last loaded values from.
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
            blend_mode: BlendMode::Normal,
            crop_left: 0.0,
            crop_top: 0.0,
            crop_right: 100.0,
            crop_bottom: 100.0,
            crop_feather: 0.0,
            volume: 100.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_ms: 0.0,
            fade_out_ms: 0.0,
            speed: 1.0,
            text_content: String::new(),
            font_size: 48.0,
            text_color: [1.0, 1.0, 1.0, 1.0],
            stroke_enabled: false,
            stroke_color: [0.0, 0.0, 0.0],
            stroke_width: 2.0,
            bg_enabled: false,
            bg_color: [0.0, 0.0, 0.0, 0.85],
            bg_padding: 12.0,
            chroma_key_enabled: false,
            chroma_key_color: [0.0, 1.0, 0.0],
            chroma_key_tolerance: 0.3,
            chroma_key_softness: 0.1,
            eyedropper_active: false,
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
    if selected_clip != state.last_loaded_clip {
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
            state.fade_in_ms = (project.clips.fade_in_pts_at(idx) as f64 * 1000.0 / 90_000.0) as f32;
            state.fade_out_ms = (project.clips.fade_out_pts_at(idx) as f64 * 1000.0 / 90_000.0) as f32;
            state.speed = project.clips.speed_at(idx);
            state.blend_mode = project.clips.blend_mode_at(idx);
            let crop = project.clips.crop_at(idx);
            state.crop_left = crop.left * 100.0;
            state.crop_top = crop.top * 100.0;
            state.crop_right = crop.right * 100.0;
            state.crop_bottom = crop.bottom * 100.0;
            state.crop_feather = crop.feather;
            // Load text clip properties
            match project.clips.kind_at(idx) {
                nexir::timeline::store::ClipKind::Text {
                    text, font_size, color, stroke_color, stroke_width, background_color, bg_padding
                } => {
                    state.text_content = text.clone();
                    state.font_size = *font_size;
                    state.text_color = *color;
                    if let Some(sc) = stroke_color {
                        state.stroke_enabled = true;
                        state.stroke_color = [sc[0], sc[1], sc[2]];
                    } else {
                        state.stroke_enabled = false;
                    }
                    state.stroke_width = *stroke_width;
                    if let Some(bg) = background_color {
                        state.bg_enabled = true;
                        state.bg_color = *bg;
                    } else {
                        state.bg_enabled = false;
                    }
                    state.bg_padding = *bg_padding;
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

                            // ── Stroke ────────────────────────────────────
                            ui.label("Stroke");
                            text_changed |= ui.checkbox(&mut state.stroke_enabled, "").changed();
                            ui.end_row();

                            if state.stroke_enabled {
                                ui.label("  Color");
                                if ui.color_edit_button_rgb(&mut state.stroke_color).changed() {
                                    text_changed = true;
                                }
                                ui.end_row();

                                ui.label("  Width");
                                text_changed |= ui
                                    .add(egui::Slider::new(&mut state.stroke_width, 0.5..=20.0).suffix("px"))
                                    .changed();
                                ui.end_row();
                            }

                            // ── Background ────────────────────────────────
                            ui.label("Background");
                            text_changed |= ui.checkbox(&mut state.bg_enabled, "").changed();
                            ui.end_row();

                            if state.bg_enabled {
                                ui.label("  Color");
                                if ui.color_edit_button_rgba_unmultiplied(&mut state.bg_color).changed() {
                                    text_changed = true;
                                }
                                ui.end_row();

                                ui.label("  Padding");
                                text_changed |= ui
                                    .add(egui::Slider::new(&mut state.bg_padding, 0.0..=100.0).suffix("px"))
                                    .changed();
                                ui.end_row();
                            }
                        });
                });
                if text_changed {
                    history.record(project);
                    project.clips.set_kind_at(idx, nexir::timeline::store::ClipKind::Text {
                        text: state.text_content.clone(),
                        font_size: state.font_size,
                        color: state.text_color,
                        stroke_color: if state.stroke_enabled {
                            Some([state.stroke_color[0], state.stroke_color[1], state.stroke_color[2], 1.0])
                        } else {
                            None
                        },
                        stroke_width: state.stroke_width,
                        background_color: if state.bg_enabled {
                            Some(state.bg_color)
                        } else {
                            None
                        },
                        bg_padding: state.bg_padding,
                    });
                }
                ui.add_space(6.0);
            }

            // ── Transform & Compositing ──────────────────────────────────
            let mut transform_changed = false;
            let mut opacity_changed = false;
            let mut blend_changed = false;
            let mut crop_changed = false;

            ui.collapsing("🎬  Transform & Compositing", |ui| {
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

                        // ── Blend Mode ────────────────────────────────────
                        ui.label("Blend Mode");
                        egui::ComboBox::from_id_source("blend_mode_dropdown")
                            .selected_text(state.blend_mode.label())
                            .show_ui(ui, |ui| {
                                for &mode in BlendMode::all() {
                                    if ui.selectable_value(&mut state.blend_mode, mode, mode.label()).clicked() {
                                        blend_changed = true;
                                    }
                                }
                            });
                        ui.end_row();
                    });

                ui.add_space(4.0);

                // ── Crop & Feathering ─────────────────────────────────────
                ui.collapsing("✂  Crop & Feather", |ui| {
                    egui::Grid::new("crop_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Left");
                            crop_changed |= ui.add(egui::Slider::new(&mut state.crop_left, 0.0..=100.0).suffix("%")).changed();
                            ui.end_row();

                            ui.label("Top");
                            crop_changed |= ui.add(egui::Slider::new(&mut state.crop_top, 0.0..=100.0).suffix("%")).changed();
                            ui.end_row();

                            ui.label("Right");
                            crop_changed |= ui.add(egui::Slider::new(&mut state.crop_right, 0.0..=100.0).suffix("%")).changed();
                            ui.end_row();

                            ui.label("Bottom");
                            crop_changed |= ui.add(egui::Slider::new(&mut state.crop_bottom, 0.0..=100.0).suffix("%")).changed();
                            ui.end_row();

                            ui.label("Feather");
                            crop_changed |= ui.add(egui::Slider::new(&mut state.crop_feather, 0.0..=1.0)).changed();
                            ui.end_row();
                        });

                    if ui.button("↺  Reset Crop").clicked() {
                        state.crop_left = 0.0;
                        state.crop_top = 0.0;
                        state.crop_right = 100.0;
                        state.crop_bottom = 100.0;
                        state.crop_feather = 0.0;
                        crop_changed = true;
                    }
                });

                ui.add_space(4.0);
                if ui.button("↺  Reset Transform").clicked() {
                    state.scale = 1.0;
                    state.pos_x = 0.0;
                    state.pos_y = 0.0;
                    state.rotation = 0.0;
                    state.opacity = 100.0;
                    state.blend_mode = BlendMode::Normal;
                    transform_changed = true;
                    opacity_changed = true;
                    blend_changed = true;
                }
            });
            // Write edited values back into the clip store
            if transform_changed || opacity_changed || blend_changed || crop_changed {
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
            if blend_changed {
                project.clips.set_blend_mode_at(idx, state.blend_mode);
            }
            if crop_changed {
                project.clips.set_crop_at(
                    idx,
                    CropRect {
                        left: state.crop_left / 100.0,
                        top: state.crop_top / 100.0,
                        right: state.crop_right / 100.0,
                        bottom: state.crop_bottom / 100.0,
                        feather: state.crop_feather,
                    },
                );
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

                        ui.label("Fade In");
                        audio_changed |= ui
                            .add(
                                egui::Slider::new(&mut state.fade_in_ms, 0.0..=5000.0)
                                    .suffix(" ms")
                                    .fixed_decimals(0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Fade Out");
                        audio_changed |= ui
                            .add(
                                egui::Slider::new(&mut state.fade_out_ms, 0.0..=5000.0)
                                    .suffix(" ms")
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
                        state.fade_in_ms = 0.0;
                        state.fade_out_ms = 0.0;
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
                // Convert ms → 90 kHz PTS ticks
                let fade_in_pts = (state.fade_in_ms as f64 * 90_000.0 / 1000.0).round() as i64;
                let fade_out_pts = (state.fade_out_ms as f64 * 90_000.0 / 1000.0).round() as i64;
                project.clips.set_fade_in_pts_at(idx, fade_in_pts);
                project.clips.set_fade_out_pts_at(idx, fade_out_pts);
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
                ui.collapsing("🟢  Chroma Key (Green Screen)", |ui| {
                    egui::Grid::new("chroma_key_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Enable");
                            ui.checkbox(&mut state.chroma_key_enabled, "");
                            ui.end_row();

                            if state.chroma_key_enabled {
                                ui.label("Key Color");
                                ui.horizontal(|ui| {
                                    ui.color_edit_button_rgb(&mut state.chroma_key_color);
                                    let eye_label = if state.eyedropper_active { "💉 Picking…" } else { "🔍 Pick" };
                                    let eye_btn = egui::Button::new(eye_label);
                                    let eye_btn = if state.eyedropper_active {
                                        eye_btn.fill(egui::Color32::from_rgb(200, 120, 0))
                                    } else {
                                        eye_btn
                                    };
                                    if ui.add(eye_btn).clicked() {
                                        state.eyedropper_active = !state.eyedropper_active;
                                    }
                                });
                                ui.end_row();

                                ui.label("Presets");
                                ui.horizontal(|ui| {
                                    if ui.button("🟩 Green").clicked() {
                                        state.chroma_key_color = [0.0, 1.0, 0.0];
                                    }
                                    if ui.button("🟦 Blue").clicked() {
                                        state.chroma_key_color = [0.0, 0.0, 1.0];
                                    }
                                });
                                ui.end_row();

                                ui.label("Tolerance");
                                ui.add(egui::Slider::new(&mut state.chroma_key_tolerance, 0.01..=1.0));
                                ui.end_row();

                                ui.label("Softness");
                                ui.add(egui::Slider::new(&mut state.chroma_key_softness, 0.0..=state.chroma_key_tolerance));
                                ui.end_row();
                            }
                        });
                });
            });
        }
    }
}
