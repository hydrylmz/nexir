// ui/src/layout/keyframe_editor.rs

use egui::{Color32, RichText, Ui};
use nexir::project::Project;
use nexir::timeline::ids::ClipId;
use nexir::timeline::keyframe::{AnimParam, InterpMode};
use crate::history::HistoryState;

/// Draws a keyframe toggle button (◆ / ◇) for a specific parameter.
/// Returns true if the keyframe state changed.
pub fn keyframe_toggle_button(
    ui: &mut Ui,
    project: &mut Project,
    clip_id: ClipId,
    param: AnimParam,
    current_pts: i64,
    current_val: f32,
    history: &mut HistoryState,
) -> bool {
    let has_key_at_now = project.clips.keyframes().has_keyframe_at(clip_id, param, current_pts);
    let has_any_keys = project.clips.keyframes().has_keyframes(clip_id, param);

    let (icon, color, tooltip) = if has_key_at_now {
        ("◆", Color32::from_rgb(0, 200, 255), "Remove keyframe at playhead")
    } else if has_any_keys {
        ("◇", Color32::from_rgb(0, 180, 230), "Add keyframe at playhead")
    } else {
        ("◇", Color32::from_rgb(120, 120, 120), "Enable keyframing (Add keyframe at playhead)")
    };

    let btn = egui::Button::new(RichText::new(icon).color(color).size(14.0))
        .frame(false);

    if ui.add(btn).on_hover_text(tooltip).clicked() {
        history.record(project);
        if has_key_at_now {
            project.clips.keyframes_mut().remove_keyframe(clip_id, param, current_pts);
        } else {
            project.clips.keyframes_mut().set_keyframe(
                clip_id,
                param,
                current_pts,
                current_val,
                InterpMode::Linear,
            );
        }
        true
    } else {
        false
    }
}

/// Helper to draw a keyframe track editor (list of keyframes with PTS, Value, InterpMode, Delete).
pub fn draw_keyframe_track_editor(
    ui: &mut Ui,
    project: &mut Project,
    clip_id: ClipId,
    param: AnimParam,
    history: &mut HistoryState,
) {
    let keys: Vec<(i64, f32, InterpMode)> = if let Some(track) = project.clips.keyframes().get_track(clip_id, param) {
        track.keys.iter().map(|k| (k.pts, k.value, k.interp)).collect()
    } else {
        Vec::new()
    };

    if keys.is_empty() {
        ui.label(RichText::new("No keyframes").color(Color32::GRAY).italics().small());
        return;
    }

    let mut action_remove = None;
    let mut action_update = None;

    egui::Grid::new(format!("kf_grid_{:?}_{:?}", clip_id, param))
        .num_columns(4)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label(RichText::new("Frame").small().strong());
            ui.label(RichText::new("Value").small().strong());
            ui.label(RichText::new("Interp").small().strong());
            ui.label("");
            ui.end_row();

            
            for (pts, val, interp) in &keys {
                let frame = project.pts_to_frame(*pts);
                ui.label(RichText::new(format!("{}", frame)).small().monospace());

                let mut val_edit = *val;
                if ui.add(egui::DragValue::new(&mut val_edit).speed(0.01)).changed() {
                    action_update = Some((*pts, val_edit, *interp));
                }

                let mut cur_interp = *interp;
                egui::ComboBox::from_id_source(format!("kf_interp_{}_{}_{}", clip_id.index(), param as u32, pts))
                    .selected_text(cur_interp.label())
                    .width(60.0)
                    .show_ui(ui, |ui| {
                        for &mode in InterpMode::all() {
                            if ui.selectable_value(&mut cur_interp, mode, mode.label()).clicked() {
                                action_update = Some((*pts, val_edit, mode));
                            }
                        }
                    });

                if ui.button(RichText::new("🗑").color(Color32::from_rgb(220, 80, 80)).small()).clicked() {
                    action_remove = Some(*pts);
                }
                ui.end_row();
            }
        });

    if let Some(pts) = action_remove {
        history.record(project);
        project.clips.keyframes_mut().remove_keyframe(clip_id, param, pts);
    }

    if let Some((pts, val, interp)) = action_update {
        history.record(project);
        project.clips.keyframes_mut().set_keyframe(clip_id, param, pts, val, interp);
    }
}
