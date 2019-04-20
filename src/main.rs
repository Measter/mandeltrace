extern crate image;
use image::{LumaA, Rgba, RgbaImage, Pixel};
extern crate imageproc;
use imageproc::drawing::draw_antialiased_line_segment_mut as draw_line;

extern crate num;
use num::complex::Complex64;

extern crate itertools;
use itertools::Itertools;

extern crate indicatif;
use indicatif::{ProgressBar, ProgressStyle};

extern crate rayon;
use rayon::prelude::*;

#[macro_use] extern crate structopt;
use structopt::StructOpt;

#[macro_use] extern crate clap;

use std::sync::Mutex;

type Image = image::ImageBuffer<LumaA<u16>, Vec<u16>>;

arg_enum!{
    #[derive(Debug, Copy, Clone)]
    enum DrawMode {
        All,
        Escaped,
        Trapped
    }
}

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(short="s", default_value="2000")]
    size: u32,

    #[structopt(short="b", default_value="2.0")]
    bounds: f64,

    #[structopt(short="d", default_value="0.01")]
    delta: f64,

    #[structopt(short="l", default_value="100")]
    limit: usize,

    #[structopt(short="z", default_value="900")]
    zoom: f64,

    #[structopt(short="r", default_value="0.4")]
    re_off: f64,

    #[structopt(short="i", default_value="0.0")]
    im_off: f64,

    #[structopt(long="chunk_len", default_value="10000")]
    chunk_len: usize,

    #[structopt(short="o", default_value="64")]
    opacity: u16,

    #[structopt(short="m", raw(possible_values="&DrawMode::variants()", case_insensitive="true"), default_value="All")]
    mode: DrawMode,

    #[structopt(long="mb")]
    overlay_mandel: bool,

    #[structopt(default_value="image.png")]
    image_name: String,

    #[structopt(short="p", default_value="2.0")]
    pow: f64,
}

fn mandelbrot(z: Complex64, (x, y): (f64, f64), args: &Args) -> Complex64 {
    z.powf(args.pow) + Complex64::new(x, y)
}

fn to_image_coord(z: Complex64, args: &Args) -> (i32, i32) {
    let pos_x = (args.size as f64/2.0) + (z.re + args.re_off) * args.zoom;
    let pos_y = (args.size as f64/2.0) + (z.im + args.im_off) * args.zoom;
    (pos_x as i32, pos_y as i32)
}

fn to_complex_coord(x: u32, y: u32, args: &Args) -> Complex64 {
    let pos_x = (x as f64 - args.size as f64/2.0)/args.zoom - args.re_off;
    let pos_y = (y as f64 - args.size as f64/2.0)/args.zoom - args.im_off;

    Complex64::new(pos_x, pos_y)
}

fn blend(mut a: LumaA<u16>, mut b: LumaA<u16>, alpha: f32) -> LumaA<u16> {
    a.data[1] = (a.data[1] as f32 * alpha) as u16;
    b.blend(&a);
    b
}

fn iterate_coordinate(coord: (f64, f64), args: &Args) -> Option<Vec<Complex64>> {
    let mut z = mandelbrot(Complex64::default(), coord, &args);
    let mut points = Vec::with_capacity(args.limit+1);
    points.push(z);

    let mut did_escape = false;
    for _ in 0..args.limit {
        z = mandelbrot(z, coord, &args);
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
        _ => None
    }
}

fn iterate_chunk(chunk: &[(&f64, &f64)], image: &Mutex<&mut Image>, args: &Args) {
    let traces: Vec<_> = chunk.iter().filter_map(|&(&x, &y)| iterate_coordinate((x, y), args)).collect();

    let ref mut image = *image.lock().expect("Lock failed");

    for t in traces {
        for w in t.windows(2) {
            draw_line(*image, to_image_coord(w[0], args), to_image_coord(w[1], args), LumaA([u16::max_value(), args.opacity]), blend);
        }
    }
}

fn to_u8_image(image: &Image, base: Option<RgbaImage>) -> RgbaImage {
    let mut out = base.unwrap_or_else(|| RgbaImage::from_pixel(image.width(), image.height(), Rgba([0, 0, 0, 255])));

    out.pixels_mut().zip(image.pixels())
        .for_each(|(o, i)| {
            let i = Rgba([255, 255, 255, (i[0] >> 8) as u8]);
            o.blend(&i);
        });

    out
}

fn main() {
    let args = Args::from_args();

    let mut canvas = Image::from_pixel(args.size, args.size, LumaA([0, u16::max_value()]));
    let mut mandel = None;

    let coords: Vec<_> = (0_u32..).map(|x| -args.bounds + x as f64 * args.delta).take_while(|&x| x < args.bounds).collect();
    let all_coords: Vec<_> = coords.iter().cartesian_product(coords.iter()).collect();

    let bar = ProgressBar::new((all_coords.len()/args.chunk_len) as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}/{eta_precise}] {wide_bar:.white} {pos:>7}/{len:7} {msg}")
        .progress_chars("█▓▒░  "));

    {
        let mut_image = Mutex::new(&mut canvas);
        all_coords.par_chunks(args.chunk_len)
            .for_each(|c| {
                iterate_chunk(c, &mut_image, &args);
                bar.inc(1);
            });
    }

    if args.overlay_mandel {
        mandel = Some(RgbaImage::from_fn(args.size, args.size, |x, y| {
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
        }));
    }

    let canvas = to_u8_image(&canvas, mandel);
    canvas.save(&args.image_name).unwrap();
}
