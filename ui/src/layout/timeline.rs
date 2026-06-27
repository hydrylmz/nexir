use egui::{Ui, RichText, Color32, Rect, Vec2, Sense, Align2};
use nexir::project::Project;
use nexir::timeline::track::TrackKind;
use nexir::timeline::mutation::ClipInsertParams;
use nexir::timeline::transform::ClipTransform;
use crate::layout::media_pool::MediaEntry;

/// Timeline UI state owned by `NexirApp`.
pub struct TimelineState {
    pub playhead_frame: i64,
    pub scroll_x: f32,
    pub zoom: f32, // pixels per frame
    pub playing: bool,
    pub last_tick: Option<std::time::Instant>,
    /// Store index of the currently selected clip, if any.
    pub selected_clip: Option<usize>,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            playhead_frame: 0,
            scroll_x: 0.0,
            zoom: 8.0,
            playing: false,
            last_tick: None,
            selected_clip: None,
        }
    }
}

pub fn draw(ui: &mut Ui, project: &mut Project, state: &mut TimelineState, dragging_item: &mut Option<MediaEntry>) {
    // ── Advance playhead if playing ───────────────────────────────────
    let fps = project.settings.frame_rate.num.max(1);
    if state.playing {
        let now = std::time::Instant::now();
        if let Some(last) = state.last_tick {
            let elapsed_secs = now.duration_since(last).as_secs_f64();
            let frames_to_advance = (elapsed_secs * fps as f64) as i64;
            if frames_to_advance > 0 {
                state.playhead_frame += frames_to_advance;
                let max_frame = project.duration_frames().max(1);
                if state.playhead_frame >= max_frame {
                    state.playhead_frame = 0; // loop
                }
                state.last_tick = Some(now);
            }
        } else {
            state.last_tick = Some(now);
        }
        ui.ctx().request_repaint(); // keep animating
    }

    // ── Header / toolbar ─────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.strong(RichText::new("Timeline").color(Color32::WHITE));
        ui.separator();

        if ui.button("⏮").on_hover_text("Go to start").clicked() {
            state.playhead_frame = 0;
            state.playing = false;
        }
        if ui.button("◀◀").on_hover_text("Step back").clicked() {
            state.playing = false;
            state.playhead_frame = (state.playhead_frame - 1).max(0);
        }
        let play_label = if state.playing { "⏸" } else { "▶" };
        let play_hint  = if state.playing { "Pause" }  else { "Play" };
        if ui.button(play_label).on_hover_text(play_hint).clicked() {
            state.playing = !state.playing;
            state.last_tick = if state.playing { Some(std::time::Instant::now()) } else { None };
        }
        if ui.button("▶▶").on_hover_text("Step forward").clicked() {
            state.playing = false;
            state.playhead_frame += 1;
        }
        if ui.button("⏭").on_hover_text("Go to end").clicked() {
            state.playing = false;
            state.playhead_frame = project.duration_frames();
        }

        ui.separator();

        // Timecode display — frame → HH:MM:SS:FF
        let fps = project.settings.frame_rate.num.max(1);
        let f = state.playhead_frame;
        let frames = f % fps;
        let secs   = (f / fps) % 60;
        let mins   = (f / (fps * 60)) % 60;
        let hours  = f / (fps * 3600);
        ui.label(
            RichText::new(format!("{:02}:{:02}:{:02}:{:02}", hours, mins, secs, frames))
                .monospace()
                .color(Color32::LIGHT_GRAY),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Zoom slider
            ui.add(
                egui::Slider::new(&mut state.zoom, 2.0..=40.0)
                    .text("Zoom")
                    .clamp_to_range(true),
            );
        });
    });

    ui.separator();

    const GUTTER_W: f32 = 110.0;
    const RULER_H:  f32 = 20.0;
    const CLIP_H_PAD: f32 = 4.0;

    let any_soloed = project.tracks.any_soloed();
    let total_frames = project.duration_frames().max(300); // show at least 10 s of space
    let fps = project.settings.frame_rate.num.max(1);

    // ── Ruler + lanes canvas ────────────────────────────────────────
    egui::ScrollArea::horizontal()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let canvas_w = total_frames as f32 * state.zoom + 60.0;

            // ── Ruler — click or drag to move playhead ─────────────
            let (ruler_rect, ruler_resp) = ui.allocate_exact_size(
                Vec2::new(canvas_w + GUTTER_W, RULER_H),
                Sense::click_and_drag(),
            );
            ui.painter().rect_filled(ruler_rect, 0.0, Color32::from_rgb(20, 20, 20));

            // Move playhead on click/drag over ruler
            if ruler_resp.dragged() || ruler_resp.clicked() {
                if let Some(pos) = ui.ctx().pointer_interact_pos() {
                    let raw_x = pos.x - ruler_rect.min.x - GUTTER_W;
                    let frame = (raw_x / state.zoom).max(0.0) as i64;
                    state.playhead_frame = frame.min(total_frames);
                    state.playing = false;
                }
            }

            // Draw tick marks every 30 frames (1 second at 30 fps)
            let tick_interval: i64 = fps;
            let start_frame = 0_i64;
            let end_frame = total_frames + tick_interval;
            let mut f_mark = start_frame;
            while f_mark <= end_frame {
                let x = GUTTER_W + f_mark as f32 * state.zoom;
                let tick_top = ruler_rect.min + egui::vec2(x, 0.0);
                let tick_bot = tick_top + egui::vec2(0.0, RULER_H * 0.5);
                ui.painter().line_segment(
                    [tick_top, tick_bot],
                    egui::Stroke::new(1.0, Color32::from_rgb(80, 80, 80)),
                );
                // Label every 5 seconds
                if f_mark % (tick_interval * 5) == 0 {
                    let secs = f_mark / fps;
                    ui.painter().text(
                        tick_top + egui::vec2(3.0, 2.0),
                        Align2::LEFT_TOP,
                        format!("{}s", secs),
                        egui::FontId::proportional(10.0),
                        Color32::from_rgb(120, 120, 120),
                    );
                }
                f_mark += tick_interval;
            }

            // ── Track rows ────────────────────────────────────────
            // We collect a pending drop here and apply it AFTER the loop
            // to avoid holding an immutable borrow of project.tracks while
            // also trying to mutate project inside the loop.
            #[derive(Default)]
            struct PendingDrop {
                track_id:    Option<nexir::timeline::ids::TrackId>,
                drop_frame:  i64,
                media_path:  Option<std::path::PathBuf>,
            }
            let mut pending_drop = PendingDrop::default();

            // Helper: determine if a file is audio-only by extension
            let is_audio_file = |path: &std::path::Path| -> bool {
                path.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e.to_lowercase().as_str(),
                        "mp3" | "wav" | "aac" | "ogg" | "flac" | "m4a" | "opus" | "wma"))
                    .unwrap_or(false)
            };

            for track in project.tracks.iter() {
                let is_active = track.is_active(any_soloed);
                let track_h = track.height_px as f32;

                let (row_rect, _) = ui.allocate_exact_size(
                    Vec2::new(canvas_w + GUTTER_W, track_h),
                    Sense::hover(),
                );

                // Track gutter (left label area)
                let gutter = Rect::from_min_size(row_rect.min, Vec2::new(GUTTER_W, track_h));
                ui.painter().rect_filled(gutter, 0.0, Color32::from_rgb(28, 28, 28));

                // Track kind indicator stripe
                let stripe_color = match track.kind {
                    TrackKind::Video => Color32::from_rgb(40, 100, 190),
                    TrackKind::Audio { .. } => Color32::from_rgb(40, 160, 90),
                    TrackKind::Text => Color32::from_rgb(190, 120, 40),
                    TrackKind::Effect => Color32::from_rgb(140, 50, 190),
                };
                let stripe = Rect::from_min_size(gutter.min, Vec2::new(3.0, track_h));
                ui.painter().rect_filled(stripe, 0.0, stripe_color);

                // Track name
                let label_color = if is_active { Color32::WHITE } else { Color32::DARK_GRAY };
                ui.painter().text(
                    gutter.min + egui::vec2(10.0, track_h * 0.5),
                    Align2::LEFT_CENTER,
                    &track.name,
                    egui::FontId::proportional(12.0),
                    label_color,
                );

                // Mute / Solo buttons (drawn manually)
                let m_rect = Rect::from_min_size(
                    gutter.min + egui::vec2(GUTTER_W - 46.0, (track_h - 16.0) * 0.5),
                    Vec2::splat(16.0),
                );
                let m_color = if track.mute { Color32::YELLOW } else { Color32::from_rgb(60, 60, 60) };
                ui.painter().rect_filled(m_rect, 3.0, m_color);
                ui.painter().text(m_rect.center(), Align2::CENTER_CENTER, "M", egui::FontId::proportional(10.0), Color32::WHITE);

                let s_rect = Rect::from_min_size(
                    gutter.min + egui::vec2(GUTTER_W - 26.0, (track_h - 16.0) * 0.5),
                    Vec2::splat(16.0),
                );
                let s_color = if track.solo { Color32::from_rgb(255, 160, 0) } else { Color32::from_rgb(60, 60, 60) };
                ui.painter().rect_filled(s_rect, 3.0, s_color);
                ui.painter().text(s_rect.center(), Align2::CENTER_CENTER, "S", egui::FontId::proportional(10.0), Color32::WHITE);

                // Lane background — also acts as drop target
                let lane = Rect::from_min_size(
                    row_rect.min + egui::vec2(GUTTER_W, 0.0),
                    Vec2::new(canvas_w, track_h),
                );

                // Highlight lane if we are dragging something over it.
                // Show red if the file type is incompatible with this track.
                let is_drop_target = dragging_item.is_some() && ui.ctx().pointer_hover_pos()
                    .map(|p| lane.contains(p))
                    .unwrap_or(false);
                let is_compatible_hover = dragging_item.as_ref().map(|item| {
                    let audio = is_audio_file(&item.path);
                    match track.kind {
                        TrackKind::Audio { .. } => audio,
                        TrackKind::Video        => !audio,
                        _                       => !audio,
                    }
                }).unwrap_or(true);
                let lane_bg = if is_drop_target && !is_compatible_hover {
                    Color32::from_rgb(75, 20, 20) // red = incompatible
                } else if is_drop_target {
                    Color32::from_rgb(30, 50, 75) // blue = compatible
                } else {
                    Color32::from_rgb(22, 22, 22)
                };
                ui.painter().rect_filled(lane, 0.0, lane_bg);

                // Subtle lane separator
                ui.painter().line_segment(
                    [lane.left_bottom(), lane.right_bottom()],
                    egui::Stroke::new(1.0, Color32::from_rgb(35, 35, 35)),
                );

                // Detect a drop: collect info, apply mutation after loop
                if is_drop_target {
                    // Check compatibility between the dragged file and the track kind
                    let is_compatible = dragging_item.as_ref().map(|item| {
                        let audio = is_audio_file(&item.path);
                        match track.kind {
                            TrackKind::Audio { .. } => audio,   // audio-only files → audio track ✓
                            TrackKind::Video        => !audio,  // video/image files → video track ✓
                            _                       => !audio,
                        }
                    }).unwrap_or(false);

                    let released = ui.input(|i| i.pointer.any_released());
                    if released {
                        if is_compatible {
                            if let Some(item) = dragging_item.take() {
                                let drop_x = ui.ctx().pointer_hover_pos()
                                    .map(|p| p.x)
                                    .unwrap_or(lane.min.x);
                                let drop_frame = ((drop_x - lane.min.x) / state.zoom).max(0.0) as i64;
                                pending_drop.track_id   = Some(track.id);
                                pending_drop.drop_frame = drop_frame;
                                pending_drop.media_path = Some(item.path.clone());
                            }
                        } else {
                            // Drop rejected — consume the drag so it doesn't linger
                            *dragging_item = None;
                        }
                    }
                }

                // ── Draw clips on this track ──────────────────────
                let clip_text_color = Color32::WHITE;

                // Track whether user clicked on empty lane space (to deselect).
                let lane_resp = ui.interact(lane, egui::Id::new(("lane", track.id)), Sense::click());
                let mut clicked_a_clip = false;

                for idx in 0..project.clips.len() {
                    if project.clips.track_id_at(idx) != track.id {
                        continue;
                    }
                    let pts_in  = project.clips.pts_in_at(idx);
                    let pts_out = project.clips.pts_out_at(idx);
                    let frame_in  = project.pts_to_frame(pts_in);
                    let frame_out = project.pts_to_frame(pts_out);

                    let x_in  = GUTTER_W + frame_in  as f32 * state.zoom;
                    let x_out = GUTTER_W + frame_out as f32 * state.zoom;
                    let clip_rect = Rect::from_min_size(
                        lane.min + egui::vec2(x_in - GUTTER_W, CLIP_H_PAD),
                        Vec2::new((x_out - x_in).max(4.0), track_h - CLIP_H_PAD * 2.0),
                    );

                    let is_selected = state.selected_clip == Some(idx);

                    // Allocate an interactive sense region for click detection
                    let clip_resp = ui.interact(
                        clip_rect,
                        egui::Id::new(("clip", idx)),
                        Sense::click(),
                    );
                    if clip_resp.clicked() {
                        state.selected_clip = Some(idx);
                        clicked_a_clip = true;
                    }

                    // Base clip color — brighten slightly on hover
                    let base_color = if clip_resp.hovered() && !is_selected {
                        Color32::from_rgb(
                            (stripe_color.r() as u16 + 25).min(255) as u8,
                            (stripe_color.g() as u16 + 25).min(255) as u8,
                            (stripe_color.b() as u16 + 25).min(255) as u8,
                        )
                    } else {
                        stripe_color
                    };

                    ui.painter().rect_filled(clip_rect, 3.0, base_color);

                    // Lighter top strip
                    let top_strip = Rect::from_min_size(clip_rect.min, Vec2::new(clip_rect.width(), 3.0));
                    ui.painter().rect_filled(top_strip, 3.0, Color32::from_rgba_unmultiplied(255, 255, 255, 40));

                    // Selection highlight: bright border + inner glow overlay
                    if is_selected {
                        // Outer selection border (cyan-white)
                        ui.painter().rect_stroke(
                            clip_rect,
                            3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(0, 200, 255)),
                        );
                        // Inner semi-transparent overlay to lighten selected clip
                        ui.painter().rect_filled(
                            clip_rect,
                            3.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 30),
                        );
                    }

                    // Clip label
                    if clip_rect.width() > 30.0 {
                        let sources = project.sources.read().unwrap();
                        let source_id = project.clips.source_id_at(idx);
                        let clip_name = sources.path(source_id)
                            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                            .unwrap_or_else(|| format!("Clip {}", idx));
                        ui.painter().text(
                            clip_rect.min + egui::vec2(5.0, clip_rect.height() * 0.5),
                            Align2::LEFT_CENTER,
                            clip_name,
                            egui::FontId::proportional(10.0),
                            clip_text_color,
                        );
                    }
                }

                // Deselect when clicking empty lane space
                if lane_resp.clicked() && !clicked_a_clip {
                    state.selected_clip = None;
                }
            }

            // ── Apply any pending clip insertion ──────────────────
            if let (Some(track_id), Some(path)) = (pending_drop.track_id, pending_drop.media_path) {
                if !project.sources.read().unwrap().path_registered(&path) {
                    let (vid_info, aud_info) = {
                        if let Ok(demuxer) = nexir::io::demuxer::Demuxer::open(&path) {
                            let project_tb = project.settings.timebase;
                            let vi = demuxer.video_stream.as_ref().map(|s| nexir::timeline::source::VideoStreamInfo {
                                width:        s.width.unwrap_or(1920),
                                height:       s.height.unwrap_or(1080),
                                frame_rate:   s.frame_rate.unwrap_or(nexir::timeline::rational::Rational { num: 30, den: 1 }),
                                pixel_fmt:    nexir::timeline::source::PixelFormat::Yuv420p,
                                color_space:  nexir::timeline::source::ColorSpace::Bt709,
                                duration_pts: project_tb.from_pts(s.duration, s.time_base),
                            });
                            let ai = demuxer.audio_stream.as_ref().map(|s| nexir::timeline::source::AudioStreamInfo {
                                sample_rate:  48000,
                                channels:     2,
                                sample_fmt:   nexir::timeline::source::SampleFormat::F32Interleaved,
                                duration_pts: project_tb.from_pts(s.duration, s.time_base),
                            });
                            (vi, ai)
                        } else {
                            (None, None)
                        }
                    };
                    project.register_source(path.clone(), vid_info, aud_info);
                }
                let source_id = project.sources.read().unwrap().id_for_path(&path);
                if let Some(source_id) = source_id {
                    let mut duration = project.frame_to_pts(fps * 5); // fallback 5 sec
                    {
                        let sources = project.sources.read().unwrap();
                        if let Ok(v) = sources.video_info(source_id) {
                            if v.duration_pts > 0 { duration = v.duration_pts; }
                        }
                        if let Ok(a) = sources.audio_info(source_id) {
                            if a.duration_pts > 0 { duration = a.duration_pts; }
                        }
                    }

                    let pts_in  = project.frame_to_pts(pending_drop.drop_frame);
                    let pts_out = pts_in + duration;
                    let _ = project.insert_clip_overwrite(ClipInsertParams {
                        track_id,
                        source_id,
                        pts_in,
                        pts_out,
                        source_in:   0,
                        layer_order: 0,
                        opacity:     1.0,
                        transform:   ClipTransform::identity(),
                    });
                }
            }

            // ── Playhead line ─────────────────────────────────────
            let ph_x = GUTTER_W + state.playhead_frame as f32 * state.zoom;
            let ph_top = ruler_rect.min + egui::vec2(ph_x, 0.0);
            let ph_bot = ph_top + egui::vec2(0.0, ruler_rect.height() + project.tracks.len() as f32 * 80.0);
            ui.painter().line_segment(
                [ph_top, ph_bot],
                egui::Stroke::new(2.0, Color32::from_rgb(255, 50, 50)),
            );
            // Playhead diamond handle
            ui.painter().circle_filled(ph_top + egui::vec2(0.0, RULER_H), 5.0, Color32::from_rgb(255, 50, 50));
        });
}
