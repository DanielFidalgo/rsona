use std::env;

use rsona::audio;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example load -- <audio file>");

    let audio = audio::load(&path).expect("failed to load audio");

    println!("Loaded audio:");
    println!("  sample rate: {} Hz", audio.sample_rate);
    println!("  channels:    {}", audio.channels);
    println!("  frames:      {}", audio.frames());
    println!("  duration:    {:.2} s", audio.duration_seconds());
}
