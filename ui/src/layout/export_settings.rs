// ui/src/layout/export_settings.rs
//
// Export Settings panel — shown before an export job starts.
// The user configures codec, quality, and hardware preference here,
// then clicks Export to kick off the job.

use nexir::export::job::{AudioCodec, Container, VideoCodec, VideoQuality};
use nexir::interop::capability::InteropCapability;

/// Persistent state for the export settings panel.
#[derive(Debug, Clone)]
pub struct ExportSettings {
    pub video_codec: VideoCodec,
    pub audio_codec: AudioCodec,
    pub container: Container,
    pub crf: u32,
    pub force_cpu: bool,
    pub cpu_preset: nexir::export::job::CpuPreset,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            container: Container::Mp4,
            crf: 23,
            force_cpu: false,
            cpu_preset: nexir::export::job::CpuPreset::Faster,
        }
    }
}

impl ExportSettings {
    /// Build a `VideoQuality` from the current panel state.
    pub fn video_quality(&self) -> VideoQuality {
        if self.video_codec.supports_crf() {
            VideoQuality::Crf(self.crf)
        } else {
            VideoQuality::TargetBitrate(200_000_000) // ProRes/VP9 default
        }
    }

    /// Auto-select a sensible container for the chosen codec.
    fn auto_container(codec: VideoCodec) -> Container {
        match codec {
            VideoCodec::ProRes => Container::Mov,
            VideoCodec::Vp9 => Container::Mkv,
            _ => Container::Mp4,
        }
    }
}

/// Draw the export settings window. Returns `true` when the user clicks Export.
///
/// `open` is a mutable bool; set it to `false` to close the window from outside.
pub fn draw(
    ctx: &egui::Context,
    open: &mut bool,
    settings: &mut ExportSettings,
    capability: &InteropCapability,
    output_path: &std::path::Path,
) -> bool {
    let mut do_export = false;

    egui::Window::new("Export Settings")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .min_width(420.0)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;

            // ── Output path ──────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Output:").strong());
                ui.label(
                    output_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("<unknown>"),
                );
            });

            ui.separator();

            // ── Video codec ───────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Video codec:")
                        .strong()
                        .line_height(Some(20.0)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_source("export_video_codec")
                        .selected_text(codec_label(settings.video_codec))
                        .show_ui(ui, |ui| {
                            for &codec in &[
                                VideoCodec::H264,
                                VideoCodec::H265,
                                VideoCodec::ProRes,
                                VideoCodec::Vp9,
                            ] {
                                if ui
                                    .selectable_label(
                                        settings.video_codec == codec,
                                        codec_label(codec),
                                    )
                                    .clicked()
                                {
                                    settings.video_codec = codec;
                                    // Auto-update container to match new codec
                                    settings.container = ExportSettings::auto_container(codec);
                                }
                            }
                        });
                });
            });

            // ── Container ─────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Container:").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // ProRes forces .mov; show read-only label instead of combo
                    if settings.video_codec == VideoCodec::ProRes {
                        ui.label("MOV (required for ProRes)");
                    } else {
                        egui::ComboBox::from_id_source("export_container")
                            .selected_text(container_label(settings.container))
                            .show_ui(ui, |ui| {
                                for &c in &[Container::Mp4, Container::Mkv] {
                                    if ui
                                        .selectable_label(
                                            settings.container == c,
                                            container_label(c),
                                        )
                                        .clicked()
                                    {
                                        settings.container = c;
                                    }
                                }
                            });
                    }
                });
            });

            // ── CRF quality slider (H.264 / H.265 only) ──────────────────────
            if settings.video_codec.supports_crf() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Quality (CRF):").strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{}", settings.crf));
                        let mut crf_f = settings.crf as f32;
                        if ui
                            .add(
                                egui::Slider::new(&mut crf_f, 0.0..=51.0)
                                    .show_value(false)
                                    .clamp_to_range(true),
                            )
                            .changed()
                        {
                            settings.crf = crf_f.round() as u32;
                        }
                        ui.label("0 = lossless  51 = worst");
                    });
                });
            }

            // ── Audio codec ───────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Audio codec:").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_source("export_audio_codec")
                        .selected_text(audio_codec_label(settings.audio_codec))
                        .show_ui(ui, |ui| {
                            for &ac in &[AudioCodec::Aac, AudioCodec::Opus, AudioCodec::Pcm] {
                                if ui
                                    .selectable_label(
                                        settings.audio_codec == ac,
                                        audio_codec_label(ac),
                                    )
                                    .clicked()
                                {
                                    settings.audio_codec = ac;
                                }
                            }
                        });
                });
            });

            ui.separator();

            // ── Hardware encode ───────────────────────────────────────────────
            let hw_label = if capability.is_available() {
                format!(
                    "Hardware: NVENC available (driver {})",
                    capability.driver_version
                )
            } else {
                "Hardware: NVENC not available — CPU encode only".to_string()
            };
            ui.label(egui::RichText::new(&hw_label).weak().italics());

            // Only show the Force CPU toggle when NVENC is actually available
            if capability.is_available() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut settings.force_cpu, "Force CPU encode (software)");
                    if settings.force_cpu {
                        ui.label(
                            egui::RichText::new("⚠ Slower — use for compatibility/quality control")
                                .color(egui::Color32::GOLD)
                                .small(),
                        );
                    }
                });
            }

            // Always show the CPU Preset selector (even if NVENC is available, they might force CPU or it might fallback)
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("CPU Preset (Speed):").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_source("export_cpu_preset")
                        .selected_text(preset_label(settings.cpu_preset))
                        .show_ui(ui, |ui| {
                            use nexir::export::job::CpuPreset::*;
                            for &p in &[
                                Ultrafast, Superfast, Veryfast, Faster, Fast, Medium, Slow, Slower,
                                Veryslow,
                            ] {
                                if ui
                                    .selectable_label(settings.cpu_preset == p, preset_label(p))
                                    .clicked()
                                {
                                    settings.cpu_preset = p;
                                }
                            }
                        });
                });
            });

            ui.separator();

            // ── Action buttons ────────────────────────────────────────────────
            ui.horizontal(|ui| {
                let export_btn = ui.add_sized(
                    [120.0, 32.0],
                    egui::Button::new(egui::RichText::new("Export").strong()),
                );
                if export_btn.clicked() {
                    do_export = true;
                    *open = false;
                }

                if ui
                    .add_sized([80.0, 32.0], egui::Button::new("Cancel"))
                    .clicked()
                {
                    *open = false;
                }
            });
        });

    do_export
}

fn codec_label(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "H.264 (AVC)",
        VideoCodec::H265 => "H.265 (HEVC)",
        VideoCodec::ProRes => "Apple ProRes 4444",
        VideoCodec::Vp9 => "VP9",
    }
}

fn container_label(c: Container) -> &'static str {
    match c {
        Container::Mp4 => "MP4",
        Container::Mkv => "MKV",
        Container::Mov => "MOV",
    }
}

fn audio_codec_label(ac: AudioCodec) -> &'static str {
    match ac {
        AudioCodec::Aac => "AAC",
        AudioCodec::Opus => "Opus",
        AudioCodec::Pcm => "PCM (lossless)",
    }
}

fn preset_label(p: nexir::export::job::CpuPreset) -> &'static str {
    use nexir::export::job::CpuPreset::*;
    match p {
        Ultrafast => "Ultrafast",
        Superfast => "Superfast",
        Veryfast => "Veryfast",
        Faster => "Faster",
        Fast => "Fast",
        Medium => "Medium",
        Slow => "Slow",
        Slower => "Slower",
        Veryslow => "Veryslow",
    }
}
