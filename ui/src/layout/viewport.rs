use egui::{Ui, RichText, Color32, Vec2, Align2, Rect, pos2};
use nexir::project::Project;
use crate::layout::timeline::TimelineState;

/// Draw the preview viewport, preserving the video's aspect ratio via letter-boxing / pillar-boxing.
/// `video_width` / `video_height` are the actual decoded frame dimensions (0 = unknown).
pub fn draw(
    ui: &mut Ui,
    preview_id: Option<egui::TextureId>,
    video_width: u32,
    video_height: u32,
    state: &mut TimelineState,
    project: &mut Project,
) -> Vec2 {
    // ── Live timecode ────────────────────────────────────────────────────
    let fps = project.settings.frame_rate.num.max(1);
    let f   = state.playhead_frame;
    let ff  = f % fps;
    let ss  = (f / fps) % 60;
    let mm  = (f / (fps * 60)) % 60;
    let hh  = f / (fps * 3600);
    let timecode = format!("{:02}:{:02}:{:02}:{:02}", hh, mm, ss, ff);

    // Top: Player info / Timecode
    ui.horizontal(|ui| {
        ui.label(RichText::new("Player").color(Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(timecode).monospace().color(Color32::LIGHT_GRAY));
        });
    });

    ui.add_space(4.0);

    // Central viewport area
    let available_size = ui.available_size();

    // Reserve space for playback controls at the bottom
    let controls_height = 40.0;
    let viewport_size = Vec2::new(available_size.x, available_size.y - controls_height);

    let (rect, _response) = ui.allocate_exact_size(viewport_size, egui::Sense::hover());

    // Always fill the background black (letter-box bars).
    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);

    if let Some(tex_id) = preview_id {
        // Compute a letter-boxed / pillar-boxed rect that preserves the video aspect ratio.
        let draw_rect = if video_width > 0 && video_height > 0 {
            let video_ar = video_width as f32 / video_height as f32;
            let panel_ar = rect.width() / rect.height();

            let (draw_w, draw_h) = if video_ar > panel_ar {
                // Video is wider than the panel — fit to width, letter-box top/bottom.
                let w = rect.width();
                let h = w / video_ar;
                (w, h)
            } else {
                // Video is taller than the panel — fit to height, pillar-box left/right.
                let h = rect.height();
                let w = h * video_ar;
                (w, h)
            };

            let center = rect.center();
            Rect::from_min_size(
                pos2(center.x - draw_w * 0.5, center.y - draw_h * 0.5),
                Vec2::new(draw_w, draw_h),
            )
        } else {
            // Dimensions unknown yet — just fill the rect.
            rect
        };

        let uv = egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        ui.painter().image(tex_id, draw_rect, uv, Color32::WHITE);
    } else {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "No Media Selected",
            egui::FontId::proportional(24.0),
            Color32::DARK_GRAY,
        );
    }

    ui.add_space(8.0);

    // ── Bottom: Playback controls ─────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center).with_main_justify(true), |ui| {
            ui.horizontal(|ui| {
                if ui.button("⏮").on_hover_text("Go to start").clicked() {
                    state.playhead_frame = 0;
                    state.playing = false;
                }
                if ui.button("⏪").on_hover_text("Step back").clicked() {
                    state.playing = false;
                    state.playhead_frame = (state.playhead_frame - 1).max(0);
                }
                let play_label = if state.playing { "⏸" } else { "▶" };
                let play_hint  = if state.playing { "Pause" } else { "Play" };
                if ui.button(play_label).on_hover_text(play_hint).clicked() {
                    state.playing = !state.playing;
                    state.last_tick = if state.playing { Some(std::time::Instant::now()) } else { None };
                }
                if ui.button("⏩").on_hover_text("Step forward").clicked() {
                    state.playing = false;
                    state.playhead_frame += 1;
                }
                if ui.button("⏭").on_hover_text("Go to end").clicked() {
                    state.playing = false;
                    state.playhead_frame = project.duration_frames();
                }
            });
        });
    });

    viewport_size
}
