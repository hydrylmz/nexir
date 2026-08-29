use nexir::export::audio_encoder::AudioMuxEncoder;
use nexir::export::job::{AudioCodec, Container, CpuPreset, ExportJob, VideoCodec, VideoQuality};
use nexir::export::muxer::Muxer;
use nexir::export::video_encoder::{VideoEncoder, VideoEncoderBackend};
use nexir::io::ffi::avutil::AVRational;
use nexir::timeline::rational::Rational;
use std::path::PathBuf;

fn main() {
    let job = ExportJob {
        output_path: PathBuf::from("test_export.mp4"),
        container: Container::Mp4,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::Aac,
        width: 1920,
        height: 1080,
        frame_rate: Rational::new(30, 1),
        quality: VideoQuality::Crf(23),
        audio_bitrate: 192000,
        render_threads: 4,
        project_tb: Rational::new(1, 90000),
        pts_in: 0,
        pts_out: 3000,
        cpu_preset: CpuPreset::Medium,
        output_color: nexir::timeline::source::ColorInfo::bt709(),
        hdr10: None,
    };

    let video_enc = VideoEncoderBackend::FfmpegCpu(VideoEncoder::open(&job).unwrap());
    let audio_enc = AudioMuxEncoder::open(&job).unwrap();

    let enc_video_tb = AVRational { num: 1, den: 30 };
    let enc_audio_tb = AVRational { num: 1, den: 48000 };

    println!("Opening muxer...");
    match Muxer::open(&job, &video_enc, &audio_enc, enc_video_tb, enc_audio_tb) {
        Ok(_) => println!("Muxer opened successfully"),
        Err(e) => println!("Muxer open failed: {:?}", e),
    }
}
