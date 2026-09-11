use crate::history::HistoryState;
use egui::{Color32, RichText, Ui};
use nexir::project::Project;
use nexir::timeline::ids::ClipId;
use nexir::timeline::keyframe::{AnimParam, InterpMode};
use nexir::timeline::transform::{BlendMode, CropRect};
use crate::layout::keyframe_editor::keyframe_toggle_button;

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
    pub color_enabled: bool,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,

    pub blur_enabled: bool,
    pub blur_radius: f32,
    pub blur_sigma: f32,

    pub sharpen_enabled: bool,
    pub sharpen_amount: f32,

    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
    pub vignette_roundness: f32,

    pub chroma_key_enabled: bool,
    pub chroma_key_color: [f32; 3],
    pub chroma_key_tolerance: f32,
    pub chroma_key_softness: f32,
    pub chroma_key_min_saturation: f32,
    pub chroma_key_spill_suppress: f32,

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
            color_enabled: false,
            brightness: 0.0,
            contrast: 100.0,
            saturation: 100.0,
            hue: 0.0,
            blur_enabled: false,
            blur_radius: 10.0,
            blur_sigma: 5.0,
            sharpen_enabled: false,
            sharpen_amount: 50.0,
            vignette_enabled: false,
            vignette_intensity: 50.0,
            vignette_radius: 75.0,
            vignette_softness: 45.0,
            vignette_roundness: 100.0,
            chroma_key_enabled: false,
            chroma_key_color: [0.0, 1.0, 0.0],
            chroma_key_tolerance: 0.3,
            chroma_key_softness: 0.1,
            chroma_key_min_saturation: 0.08,
            chroma_key_spill_suppress: 0.3,
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
    playhead_pts: i64,
) {
    ui.heading(RichText::new("Inspector").color(Color32::WHITE));
    ui.separator();
    ui.add_space(4.0);

    // Wrap everything in a scroll area so content is always reachable
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            draw_inner(ui, state, project, selected_clip, history, playhead_pts);
        });
}

fn draw_inner(
    ui: &mut Ui,
    state: &mut InspectorState,
    project: &mut Project,
    selected_clip: Option<usize>,
    history: &mut HistoryState,
    playhead_pts: i64,
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

            let eff = project.clips.effects_at(idx);
            state.color_enabled = eff.color_enabled;
            state.brightness = eff.brightness * 100.0;
            state.contrast = eff.contrast * 100.0;
            state.saturation = eff.saturation * 100.0;
            state.hue = eff.hue;
            state.blur_enabled = eff.blur_enabled;
            state.blur_radius = eff.blur_radius;
            state.blur_sigma = eff.blur_sigma;
            state.sharpen_enabled = eff.sharpen_enabled;
            state.sharpen_amount = eff.sharpen_amount * 100.0;
            state.vignette_enabled = eff.vignette_enabled;
            state.vignette_intensity = eff.vignette_intensity * 100.0;
            state.vignette_radius = eff.vignette_radius * 100.0;
            state.vignette_softness = eff.vignette_softness * 100.0;
            state.vignette_roundness = eff.vignette_roundness * 100.0;
            state.chroma_key_enabled = eff.chroma_key_enabled;
            state.chroma_key_color = eff.chroma_key_color;
            state.chroma_key_tolerance = eff.chroma_key_tolerance;
            state.chroma_key_softness = eff.chroma_key_softness;
            state.chroma_key_min_saturation = eff.chroma_key_min_saturation;
            state.chroma_key_spill_suppress = eff.chroma_key_spill_suppress;

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

    // Sync external effect changes (e.g. from timeline drag) and evaluated keyframe values at the current playhead
    if let Some(idx) = selected_clip {
        let eff = project.clips.effects_at(idx);
        state.color_enabled = eff.color_enabled;
        state.blur_enabled = eff.blur_enabled;
        state.sharpen_enabled = eff.sharpen_enabled;
        state.vignette_enabled = eff.vignette_enabled;
        state.chroma_key_enabled = eff.chroma_key_enabled;

        let clip_id = project.clips.clip_id_at(idx);
        let kf = project.clips.keyframes();
        if let Some(v) = kf.eval(clip_id, AnimParam::Scale, playhead_pts) { state.scale = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::PositionX, playhead_pts) { state.pos_x = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::PositionY, playhead_pts) { state.pos_y = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Rotation, playhead_pts) { state.rotation = v.to_degrees(); }
        if let Some(v) = kf.eval(clip_id, AnimParam::Opacity, playhead_pts) { state.opacity = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::CropLeft, playhead_pts) { state.crop_left = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::CropTop, playhead_pts) { state.crop_top = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::CropRight, playhead_pts) { state.crop_right = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::CropBottom, playhead_pts) { state.crop_bottom = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::CropFeather, playhead_pts) { state.crop_feather = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Brightness, playhead_pts) { state.brightness = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Contrast, playhead_pts) { state.contrast = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Saturation, playhead_pts) { state.saturation = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::HueShift, playhead_pts) { state.hue = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::BlurRadius, playhead_pts) { state.blur_radius = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::BlurSigma, playhead_pts) { state.blur_sigma = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::SharpenAmount, playhead_pts) { state.sharpen_amount = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::VignetteIntensity, playhead_pts) { state.vignette_intensity = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::VignetteRadius, playhead_pts) { state.vignette_radius = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::VignetteSoftness, playhead_pts) { state.vignette_softness = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::VignetteRoundness, playhead_pts) { state.vignette_roundness = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::ChromaKeyTolerance, playhead_pts) { state.chroma_key_tolerance = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::ChromaKeySoftness, playhead_pts) { state.chroma_key_softness = v; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Volume, playhead_pts) { state.volume = v * 100.0; }
        if let Some(v) = kf.eval(clip_id, AnimParam::Pan, playhead_pts) { state.pan = v * 100.0; }
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
            let clip_id = project.clips.clip_id_at(idx);
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
                        ui.horizontal(|ui| {
                            transform_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Scale, playhead_pts, state.scale, history);
                            ui.label("Scale");
                        });
                        let scale_resp = ui.add(egui::Slider::new(&mut state.scale, 0.1..=5.0).suffix("x"));
                        if scale_resp.changed() {
                            transform_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Scale) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Scale, playhead_pts, state.scale, InterpMode::Linear);
                            }
                        }
                        ui.end_row();

                        ui.horizontal(|ui| {
                            transform_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::PositionX, playhead_pts, state.pos_x, history);
                            ui.label("Position X");
                        });
                        let px_resp = ui.add(egui::Slider::new(&mut state.pos_x, -(project.settings.width as f32)..=(project.settings.width as f32)).suffix("px"));
                        if px_resp.changed() {
                            transform_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::PositionX) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::PositionX, playhead_pts, state.pos_x, InterpMode::Linear);
                            }
                        }
                        ui.end_row();

                        ui.horizontal(|ui| {
                            transform_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::PositionY, playhead_pts, state.pos_y, history);
                            ui.label("Position Y");
                        });
                        let py_resp = ui.add(egui::Slider::new(&mut state.pos_y, -(project.settings.height as f32)..=(project.settings.height as f32)).suffix("px"));
                        if py_resp.changed() {
                            transform_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::PositionY) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::PositionY, playhead_pts, state.pos_y, InterpMode::Linear);
                            }
                        }
                        ui.end_row();

                        ui.horizontal(|ui| {
                            transform_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Rotation, playhead_pts, state.rotation.to_radians(), history);
                            ui.label("Rotation");
                        });
                        let rot_resp = ui.add(egui::Slider::new(&mut state.rotation, -180.0..=180.0).suffix("°"));
                        if rot_resp.changed() {
                            transform_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Rotation) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Rotation, playhead_pts, state.rotation.to_radians(), InterpMode::Linear);
                            }
                        }
                        ui.end_row();

                        ui.horizontal(|ui| {
                            opacity_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Opacity, playhead_pts, state.opacity / 100.0, history);
                            ui.label("Opacity");
                        });
                        let op_resp = ui.add(egui::Slider::new(&mut state.opacity, 0.0..=100.0).suffix("%"));
                        if op_resp.changed() {
                            opacity_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Opacity) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Opacity, playhead_pts, state.opacity / 100.0, InterpMode::Linear);
                            }
                        }
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
                            ui.horizontal(|ui| {
                                crop_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::CropLeft, playhead_pts, state.crop_left / 100.0, history);
                                ui.label("Left");
                            });
                            let cl_resp = ui.add(egui::Slider::new(&mut state.crop_left, 0.0..=100.0).suffix("%"));
                            if cl_resp.changed() {
                                crop_changed = true;
                                if project.clips.keyframes().has_keyframes(clip_id, AnimParam::CropLeft) {
                                    history.record(project);
                                    project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::CropLeft, playhead_pts, state.crop_left / 100.0, InterpMode::Linear);
                                }
                            }
                            ui.end_row();

                            ui.horizontal(|ui| {
                                crop_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::CropTop, playhead_pts, state.crop_top / 100.0, history);
                                ui.label("Top");
                            });
                            let ct_resp = ui.add(egui::Slider::new(&mut state.crop_top, 0.0..=100.0).suffix("%"));
                            if ct_resp.changed() {
                                crop_changed = true;
                                if project.clips.keyframes().has_keyframes(clip_id, AnimParam::CropTop) {
                                    history.record(project);
                                    project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::CropTop, playhead_pts, state.crop_top / 100.0, InterpMode::Linear);
                                }
                            }
                            ui.end_row();

                            ui.horizontal(|ui| {
                                crop_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::CropRight, playhead_pts, state.crop_right / 100.0, history);
                                ui.label("Right");
                            });
                            let cr_resp = ui.add(egui::Slider::new(&mut state.crop_right, 0.0..=100.0).suffix("%"));
                            if cr_resp.changed() {
                                crop_changed = true;
                                if project.clips.keyframes().has_keyframes(clip_id, AnimParam::CropRight) {
                                    history.record(project);
                                    project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::CropRight, playhead_pts, state.crop_right / 100.0, InterpMode::Linear);
                                }
                            }
                            ui.end_row();

                            ui.horizontal(|ui| {
                                crop_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::CropBottom, playhead_pts, state.crop_bottom / 100.0, history);
                                ui.label("Bottom");
                            });
                            let cb_resp = ui.add(egui::Slider::new(&mut state.crop_bottom, 0.0..=100.0).suffix("%"));
                            if cb_resp.changed() {
                                crop_changed = true;
                                if project.clips.keyframes().has_keyframes(clip_id, AnimParam::CropBottom) {
                                    history.record(project);
                                    project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::CropBottom, playhead_pts, state.crop_bottom / 100.0, InterpMode::Linear);
                                }
                            }
                            ui.end_row();

                            ui.horizontal(|ui| {
                                crop_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::CropFeather, playhead_pts, state.crop_feather, history);
                                ui.label("Feather");
                            });
                            let cf_resp = ui.add(egui::Slider::new(&mut state.crop_feather, 0.0..=1.0));
                            if cf_resp.changed() {
                                crop_changed = true;
                                if project.clips.keyframes().has_keyframes(clip_id, AnimParam::CropFeather) {
                                    history.record(project);
                                    project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::CropFeather, playhead_pts, state.crop_feather, InterpMode::Linear);
                                }
                            }
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

                        ui.horizontal(|ui| {
                            audio_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Volume, playhead_pts, state.volume / 100.0, history);
                            ui.label("Volume");
                        });
                        let vol_resp = ui.add(
                            egui::Slider::new(&mut state.volume, 0.0..=200.0)
                                .suffix("%")
                                .fixed_decimals(0),
                        );
                        if vol_resp.changed() {
                            audio_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Volume) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Volume, playhead_pts, state.volume / 100.0, InterpMode::Linear);
                            }
                        }
                        ui.end_row();

                        ui.horizontal(|ui| {
                            audio_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Pan, playhead_pts, state.pan / 100.0, history);
                            ui.label("Pan");
                        });
                        let pan_resp = ui.add(
                            egui::Slider::new(&mut state.pan, -100.0..=100.0)
                                .suffix("%")
                                .fixed_decimals(0),
                        );
                        if pan_resp.changed() {
                            audio_changed = true;
                            if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Pan) {
                                history.record(project);
                                project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Pan, playhead_pts, state.pan / 100.0, InterpMode::Linear);
                            }
                        }
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
                                    && let Some(linked_idx) = project.clips.index_of(linked_id) {
                                        project.clips.set_speed_at(linked_idx, state.speed);
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
                                    && let Some(linked_idx) = project.clips.index_of(linked_id) {
                                        project.clips.set_speed_at(linked_idx, preset);
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
            let any_effects = state.color_enabled
                || state.blur_enabled
                || state.sharpen_enabled
                || state.vignette_enabled
                || state.chroma_key_enabled;

            let mut effects_changed = false;

            ui.collapsing("✨  Effects", |ui| {
                if !any_effects {
                    ui.add_space(4.0);
                    egui::Frame::none()
                        .fill(Color32::from_rgb(24, 24, 24))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(45, 45, 45)))
                        .rounding(6.0)
                        .inner_margin(egui::Margin::symmetric(12.0, 12.0))
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.vertical_centered(|ui| {
                                ui.label(RichText::new("✨").size(24.0));
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new("No effects applied")
                                        .color(Color32::from_rgb(160, 160, 160))
                                        .strong()
                                        .size(12.0),
                                );
                                ui.add_space(2.0);
                                ui.label(
                                    RichText::new("Drag effects from the Effects tab on the left onto this clip.")
                                        .color(Color32::from_rgb(110, 110, 110))
                                        .size(11.0),
                                );
                            });
                        });
                } else {
                    // 1. Color & Light
                    if state.color_enabled {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("🎨  Color & Light").strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80))).on_hover_text("Remove effect").clicked() {
                                    state.color_enabled = false;
                                    effects_changed = true;
                                }
                            });
                        });
                        egui::Grid::new("color_light_grid")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Brightness, playhead_pts, state.brightness / 100.0, history);
                                    ui.label("Brightness");
                                });
                                let br_resp = ui.add(egui::Slider::new(&mut state.brightness, -100.0..=100.0).suffix("%"));
                                if br_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Brightness) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Brightness, playhead_pts, state.brightness / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Contrast, playhead_pts, state.contrast / 100.0, history);
                                    ui.label("Contrast");
                                });
                                let ct_resp = ui.add(egui::Slider::new(&mut state.contrast, 0.0..=200.0).suffix("%"));
                                if ct_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Contrast) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Contrast, playhead_pts, state.contrast / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::Saturation, playhead_pts, state.saturation / 100.0, history);
                                    ui.label("Saturation");
                                });
                                let st_resp = ui.add(egui::Slider::new(&mut state.saturation, 0.0..=200.0).suffix("%"));
                                if st_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::Saturation) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::Saturation, playhead_pts, state.saturation / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::HueShift, playhead_pts, state.hue, history);
                                    ui.label("Hue Shift");
                                });
                                let hue_resp = ui.add(egui::Slider::new(&mut state.hue, -180.0..=180.0).suffix("°"));
                                if hue_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::HueShift) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::HueShift, playhead_pts, state.hue, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();
                            });

                        ui.add_space(2.0);
                        if ui.button("↺  Reset Color").clicked() {
                            state.brightness = 0.0;
                            state.contrast = 100.0;
                            state.saturation = 100.0;
                            state.hue = 0.0;
                            effects_changed = true;
                        }
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);
                    }

                    // 2. Gaussian Blur
                    if state.blur_enabled {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("💧  Gaussian Blur").strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80))).on_hover_text("Remove effect").clicked() {
                                    state.blur_enabled = false;
                                    effects_changed = true;
                                }
                            });
                        });
                        egui::Grid::new("blur_grid")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::BlurRadius, playhead_pts, state.blur_radius, history);
                                    ui.label("Radius");
                                });
                                let rad_resp = ui.add(egui::Slider::new(&mut state.blur_radius, 1.0..=50.0).suffix(" px"));
                                if rad_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::BlurRadius) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::BlurRadius, playhead_pts, state.blur_radius, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::BlurSigma, playhead_pts, state.blur_sigma, history);
                                    ui.label("Sigma");
                                });
                                let sig_resp = ui.add(egui::Slider::new(&mut state.blur_sigma, 0.5..=25.0));
                                if sig_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::BlurSigma) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::BlurSigma, playhead_pts, state.blur_sigma, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);
                    }

                    // 3. Sharpen
                    if state.sharpen_enabled {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("⚡  Sharpen").strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80))).on_hover_text("Remove effect").clicked() {
                                    state.sharpen_enabled = false;
                                    effects_changed = true;
                                }
                            });
                        });
                        egui::Grid::new("sharpen_grid")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::SharpenAmount, playhead_pts, state.sharpen_amount / 100.0, history);
                                    ui.label("Amount");
                                });
                                let sh_resp = ui.add(egui::Slider::new(&mut state.sharpen_amount, 0.0..=200.0).suffix("%"));
                                if sh_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::SharpenAmount) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::SharpenAmount, playhead_pts, state.sharpen_amount / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);
                    }

                    // 4. Vignette
                    if state.vignette_enabled {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("🌑  Vignette").strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80))).on_hover_text("Remove effect").clicked() {
                                    state.vignette_enabled = false;
                                    effects_changed = true;
                                }
                            });
                        });
                        egui::Grid::new("vignette_grid")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::VignetteIntensity, playhead_pts, state.vignette_intensity / 100.0, history);
                                    ui.label("Intensity");
                                });
                                let vi_resp = ui.add(egui::Slider::new(&mut state.vignette_intensity, 0.0..=100.0).suffix("%"));
                                if vi_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::VignetteIntensity) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::VignetteIntensity, playhead_pts, state.vignette_intensity / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::VignetteRadius, playhead_pts, state.vignette_radius / 100.0, history);
                                    ui.label("Radius");
                                });
                                let vr_resp = ui.add(egui::Slider::new(&mut state.vignette_radius, 10.0..=150.0).suffix("%"));
                                if vr_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::VignetteRadius) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::VignetteRadius, playhead_pts, state.vignette_radius / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::VignetteSoftness, playhead_pts, state.vignette_softness / 100.0, history);
                                    ui.label("Softness");
                                });
                                let vs_resp = ui.add(egui::Slider::new(&mut state.vignette_softness, 0.0..=100.0).suffix("%"));
                                if vs_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::VignetteSoftness) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::VignetteSoftness, playhead_pts, state.vignette_softness / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::VignetteRoundness, playhead_pts, state.vignette_roundness / 100.0, history);
                                    ui.label("Roundness");
                                });
                                let vrnd_resp = ui.add(egui::Slider::new(&mut state.vignette_roundness, 0.0..=100.0).suffix("%"));
                                if vrnd_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::VignetteRoundness) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::VignetteRoundness, playhead_pts, state.vignette_roundness / 100.0, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);
                    }

                    // 5. Chroma Key
                    if state.chroma_key_enabled {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("🟢  Chroma Key").strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80))).on_hover_text("Remove effect").clicked() {
                                    state.chroma_key_enabled = false;
                                    effects_changed = true;
                                }
                            });
                        });
                        egui::Grid::new("chroma_key_grid")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Key Color");
                                ui.horizontal(|ui| {
                                    effects_changed |= ui.color_edit_button_rgb(&mut state.chroma_key_color).changed();
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
                                        effects_changed = true;
                                    }
                                    if ui.button("🟦 Blue").clicked() {
                                        state.chroma_key_color = [0.0, 0.0, 1.0];
                                        effects_changed = true;
                                    }
                                });
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::ChromaKeyTolerance, playhead_pts, state.chroma_key_tolerance, history);
                                    ui.label("Tolerance");
                                });
                                let tol_resp = ui.add(egui::Slider::new(&mut state.chroma_key_tolerance, 0.01..=1.0));
                                if tol_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::ChromaKeyTolerance) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::ChromaKeyTolerance, playhead_pts, state.chroma_key_tolerance, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.horizontal(|ui| {
                                    effects_changed |= keyframe_toggle_button(ui, project, clip_id, AnimParam::ChromaKeySoftness, playhead_pts, state.chroma_key_softness, history);
                                    ui.label("Softness");
                                });
                                let sft_resp = ui.add(egui::Slider::new(&mut state.chroma_key_softness, 0.0..=state.chroma_key_tolerance));
                                if sft_resp.changed() {
                                    effects_changed = true;
                                    if project.clips.keyframes().has_keyframes(clip_id, AnimParam::ChromaKeySoftness) {
                                        history.record(project);
                                        project.clips.keyframes_mut().set_keyframe(clip_id, AnimParam::ChromaKeySoftness, playhead_pts, state.chroma_key_softness, InterpMode::Linear);
                                    }
                                }
                                ui.end_row();

                                ui.label("Spill Suppress");
                                let sp_resp = ui.add(egui::Slider::new(&mut state.chroma_key_spill_suppress, 0.0..=1.0).custom_formatter(|n, _| format!("{:.0}%", n * 100.0)));
                                if sp_resp.changed() {
                                    effects_changed = true;
                                }
                                ui.end_row();

                                ui.label("Min Saturation");
                                let sat_resp = ui.add(egui::Slider::new(&mut state.chroma_key_min_saturation, 0.0..=1.0).custom_formatter(|n, _| format!("{:.0}%", n * 100.0)));
                                if sat_resp.changed() {
                                    effects_changed = true;
                                }
                                ui.end_row();
                            });
                    }
                }
            });

            if effects_changed {
                history.record(project);
                let eff = nexir::timeline::transform::ClipEffects {
                    color_enabled: state.color_enabled,
                    brightness: state.brightness / 100.0,
                    contrast: state.contrast / 100.0,
                    saturation: state.saturation / 100.0,
                    hue: state.hue,
                    blur_enabled: state.blur_enabled,
                    blur_radius: state.blur_radius,
                    blur_sigma: state.blur_sigma,
                    sharpen_enabled: state.sharpen_enabled,
                    sharpen_amount: state.sharpen_amount / 100.0,
                    vignette_enabled: state.vignette_enabled,
                    vignette_intensity: state.vignette_intensity / 100.0,
                    vignette_radius: state.vignette_radius / 100.0,
                    vignette_softness: state.vignette_softness / 100.0,
                    vignette_roundness: state.vignette_roundness / 100.0,
                    chroma_key_enabled: state.chroma_key_enabled,
                    chroma_key_color: state.chroma_key_color,
                    chroma_key_tolerance: state.chroma_key_tolerance,
                    chroma_key_softness: state.chroma_key_softness,
                    chroma_key_min_saturation: state.chroma_key_min_saturation,
                    chroma_key_spill_suppress: state.chroma_key_spill_suppress,
                };
                project.clips.set_effects_at(idx, eff);
            }

            // ── Keyframe Tracks Section ───────────────────────────────────
            if project.clips.keyframes().has_any_keyframes_for_clip(clip_id) {
                ui.add_space(6.0);
                ui.collapsing("◆  Keyframe Tracks", |ui| {
                    for param in [
                        AnimParam::Scale,
                        AnimParam::PositionX,
                        AnimParam::PositionY,
                        AnimParam::Rotation,
                        AnimParam::Opacity,
                        AnimParam::CropLeft,
                        AnimParam::CropTop,
                        AnimParam::CropRight,
                        AnimParam::CropBottom,
                        AnimParam::CropFeather,
                        AnimParam::Brightness,
                        AnimParam::Contrast,
                        AnimParam::Saturation,
                        AnimParam::HueShift,
                        AnimParam::BlurRadius,
                        AnimParam::BlurSigma,
                        AnimParam::SharpenAmount,
                        AnimParam::VignetteIntensity,
                        AnimParam::VignetteRadius,
                        AnimParam::VignetteSoftness,
                        AnimParam::VignetteRoundness,
                        AnimParam::ChromaKeyTolerance,
                        AnimParam::ChromaKeySoftness,
                        AnimParam::Volume,
                        AnimParam::Pan,
                    ] {
                        if project.clips.keyframes().has_keyframes(clip_id, param) {
                            ui.label(RichText::new(param.label()).strong().small().color(Color32::from_rgb(0, 200, 255)));
                            crate::layout::keyframe_editor::draw_keyframe_track_editor(ui, project, clip_id, param, history);
                            ui.add_space(4.0);
                        }
                    }
                });
            }
        }
    }
}
