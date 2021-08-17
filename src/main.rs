use image::{LumaA, Pixel, Rgba, RgbaImage};
use imageproc::drawing::draw_antialiased_line_segment_mut as draw_line;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use num::complex::Complex64;
use rayon::prelude::*;
use structopt::StructOpt;

use std::{convert::TryInto, str::FromStr};

type Image = image::ImageBuffer<LumaA<u16>, Vec<u16>>;

#[derive(Debug, Copy, Clone)]
enum DrawMode {
    All,
    Escaped,
    Trapped,
}

impl FromStr for DrawMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("all") {
            Ok(Self::All)
        } else if s.eq_ignore_ascii_case("escaped") {
            Ok(Self::Escaped)
        } else if s.eq_ignore_ascii_case("trapped") {
            Ok(Self::Trapped)
        } else {
            Err(format!("Unknown draw mode: '{}'", s))
        }
    }
}

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(short = "s", default_value = "2000")]
    size: u32,

    #[structopt(short = "b", default_value = "2.0")]
    bounds: f64,

    #[structopt(short = "d", default_value = "0.01")]
    delta: f64,

    #[structopt(short = "l", default_value = "100")]
    limit: usize,

    #[structopt(short = "z", default_value = "900")]
    zoom: f64,

    #[structopt(short = "r", default_value = "0.4")]
    re_off: f64,

    #[structopt(short = "i", default_value = "0.0")]
    im_off: f64,

    #[structopt(long = "chunk_len", default_value = "50000")]
    chunk_len: usize,

    #[structopt(short = "o", default_value = "64")]
    opacity: u16,

    #[structopt(short = "m", default_value = "All")]
    mode: DrawMode,

    #[structopt(long = "mb")]
    overlay_mandel: bool,

    #[structopt(default_value = "image.png")]
    image_name: String,

    #[structopt(short = "p", default_value = "2.0")]
    pow: f64,
}

struct ArrWindows<'a, T, const N: usize>(&'a [T]);
impl<'a, T, const N: usize> Iterator for ArrWindows<'a, T, N> {
    type Item = &'a [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.0.get(..N)?;
        self.0 = self.0.get(1..)?;
        next.try_into().ok()
    }
}

fn mandelbrot(z: Complex64, (x, y): (f64, f64), args: &Args) -> Complex64 {
    z.powf(args.pow) + Complex64::new(x, y)
}

fn to_image_coord(z: Complex64, args: &Args) -> (i32, i32) {
    let pos_x = (args.size as f64 / 2.0) + (z.re + args.re_off) * args.zoom;
    let pos_y = (args.size as f64 / 2.0) + (z.im + args.im_off) * args.zoom;
    (pos_x as i32, pos_y as i32)
}

fn to_complex_coord(x: u32, y: u32, args: &Args) -> Complex64 {
    let pos_x = (x as f64 - args.size as f64 / 2.0) / args.zoom - args.re_off;
    let pos_y = (y as f64 - args.size as f64 / 2.0) / args.zoom - args.im_off;

    Complex64::new(pos_x, pos_y)
}

fn blend(mut a: LumaA<u16>, mut b: LumaA<u16>, alpha: f32) -> LumaA<u16> {
    a.0[1] = (a.0[1] as f32 * alpha) as u16;
    b.blend(&a);
    b
}

fn iterate_coordinate(coord: (f64, f64), args: &Args) -> Option<Vec<Complex64>> {
    let mut z = mandelbrot(Complex64::default(), coord, args);
    let mut points = Vec::with_capacity(args.limit + 1);
    points.push(z);

    let mut did_escape = false;
    for _ in 0..args.limit {
        z = mandelbrot(z, coord, args);
        points.push(z);

        if z.im.abs() > args.bounds || z.re.abs() > args.bounds {
            did_escape = true;
            break;
        }
    }

    use DrawMode::*;
    match (args.mode, did_escape) {
        (All, _) => Some(points),
        (Escaped, true) => Some(points),
        (Trapped, false) => Some(points),
        _ => None,
    }
}

fn iterate_chunk(chunk: &[(&f64, &f64)], mut image: Image, args: &Args) -> Image {
    let traces = chunk
        .iter()
        .filter_map(|&(&x, &y)| iterate_coordinate((x, y), args));

    for t in traces {
        for &[w1, w2] in ArrWindows(&t) {
            draw_line(
                &mut image,
                to_image_coord(w1, args),
                to_image_coord(w2, args),
                LumaA([u16::max_value(), args.opacity]),
                blend,
            );
        }
    }

    image
}

fn to_u8_image(image: &Image, base: Option<RgbaImage>) -> RgbaImage {
    let mut out = base.unwrap_or_else(|| {
        RgbaImage::from_pixel(image.width(), image.height(), Rgba([0, 0, 0, 255]))
    });

    out.pixels_mut().zip(image.pixels()).for_each(|(o, i)| {
        let i = Rgba([255, 255, 255, (i[0] >> 8) as u8]);
        o.blend(&i);
    });

    out
}

fn main() {
    let args = Args::from_args();

    let canvas = Image::from_pixel(args.size, args.size, LumaA([0, 0]));

    let coords: Vec<_> = (0_u32..)
        .map(|x| -args.bounds + x as f64 * args.delta)
        .take_while(|&x| x < args.bounds)
        .collect();
    let all_coords: Vec<_> = coords.iter().cartesian_product(coords.iter()).collect();

    let bar = ProgressBar::new((all_coords.len() / args.chunk_len) as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}/{eta_precise}] {wide_bar:.white} {pos:>7}/{len:7} {msg}")
            .progress_chars("█▓▒░  "),
    );

    let canvas = all_coords
        .par_chunks(args.chunk_len)
        .map(|c| {
            let chunk = iterate_chunk(c, canvas.clone(), &args);
            bar.inc(1);
            chunk
        })
        .reduce(
            || canvas.clone(),
            |mut blend, chunk| {
                blend
                    .pixels_mut()
                    .zip(chunk.pixels())
                    .for_each(|(o, i)| o.blend(i));

                blend
            },
        );

    let mut background = Image::from_pixel(args.size, args.size, LumaA([0, u16::MAX]));
    background
        .pixels_mut()
        .zip(canvas.pixels())
        .for_each(|(o, i)| o.blend(i));

    let mandel = args.overlay_mandel.then(|| {
        RgbaImage::from_fn(args.size, args.size, |x, y| {
            let cmpl = to_complex_coord(x, y, &args);

            let mut z = Complex64::default();
            let mut did_escape = false;
            for _ in 0..args.limit {
                z = mandelbrot(z, (cmpl.re, cmpl.im), &args);

                if z.norm_sqr() > 4.0 {
                    did_escape = true;
                    break;
                }
            }

            if did_escape {
                Rgba([128, 0, 0, 255])
            } else {
                Rgba([0, 0, 0, 255])
            }
        })
    });

    let canvas = to_u8_image(&background, mandel);
    canvas.save(&args.image_name).unwrap();
}
