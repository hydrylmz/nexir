use std::env;
use std::path::Path;
use image::io::Reader as ImageReader;
use image::GenericImageView;

fn main() {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run --example probe_image -- <path-to-image>");
            std::process::exit(2);
        }
    };
    let p = Path::new(&path);
    println!("Probing image: {:?}", p);

    match ImageReader::open(p) {
        Ok(r) => match r.with_guessed_format() {
            Ok(r2) => match r2.decode() {
                Ok(img) => {
                    let (w, h) = img.dimensions();
                    println!("Decoded: {}x{} color: {:?}", w, h, img.color());
                    let rgba = img.to_rgba8();
                    let mut min_a = 255u8;
                    let mut max_a = 0u8;
                    for p in rgba.pixels() {
                        let a = p.0[3];
                        if a < min_a { min_a = a; }
                        if a > max_a { max_a = a; }
                    }
                    println!("Alpha min={} max={}", min_a, max_a);
                }
                Err(e) => {
                    eprintln!("Failed to decode: {:?}", e);
                    std::process::exit(3);
                }
            },
            Err(e) => {
                eprintln!("Failed to guess format: {:?}", e);
                std::process::exit(4);
            }
        },
        Err(e) => {
            eprintln!("Failed to open: {:?}", e);
            std::process::exit(5);
        }
    }
}
