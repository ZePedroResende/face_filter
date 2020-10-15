use opencv::core;
use opencv::core::Mat;
use opencv::highgui;
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc;
use opencv::objdetect::{CascadeClassifier, CASCADE_SCALE_IMAGE};
use opencv::prelude::*;
use opencv::types;
use opencv::videoio::{VideoCapture, CAP_ANY};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn put_moustache(mut fc: Mat, x: i32, y: i32, w: i32, h: i32) -> Mat {
    let mst = imread("utils/moustache.png", IMREAD_COLOR).unwrap();

    let face_width = w as f64;
    let face_height = h as f64;

    let mst_width = (face_width * 0.4166666) as i32 + 1;
    let mst_height = (face_height * 0.142857) as i32 + 1;

    let mut mst_out = Mat::default().unwrap();
    match imgproc::resize(
        &mst,
        &mut mst_out,
        core::Size {
            width: mst_width,
            height: mst_height,
        },
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    ) {
        Ok(v) => v,
        Err(e) => panic!("{}", e),
    }

    for i in
        ((0.62857142857 * face_height) as i32)..((0.62857142857 * face_height) as i32 + mst_height)
    {
        for j in
            (0.29166666666 * face_width) as i32..(0.29166666666 * face_width) as i32 + mst_width
        {
            for k in 0..3 {
                let v = match mst_out
                    .at_2d::<core::Vec3b>(
                        i - (0.62857142857 * face_height) as i32,
                        j - (0.29166666666 * face_width) as i32,
                    )
                    //.at(k)
                {
                    Ok(v) => v,
                    Err(e) => panic!("{}", e),
                };
                let v2 = v[k];

                if v2 < 235 {
                    let fc_mut = fc.at_2d_mut::<core::Vec3b>(y + i, x + j).unwrap();
                    fc_mut[k] = v2;
                }
            }
        }
    }

    fc
}

fn main() {
    let casc_path = "utils/haarcascade_frontalface_default.xml";
    if Path::new(casc_path).exists() {
        Command::new("sh")
            .arg("-c")
            .arg("utils/download_filters.sh")
            .output()
            .expect("failed to execute download of cascade classidifier ");
    };

    let mut face_cascade = CascadeClassifier::new(&casc_path).unwrap();
    let mut video_capture = VideoCapture::new(0, CAP_ANY).unwrap();

    loop {
        if !video_capture.is_opened().unwrap() {
            println!("No camera found");
            thread::sleep(Duration::from_millis(4000));
            continue;
        }

        let mut frame = Mat::default().unwrap();
        video_capture.read(&mut frame).unwrap();

        if frame.size().unwrap().width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }

        let mut gray = Mat::default().unwrap();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        let mut faces = types::VectorOfRect::new();
        match face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            2,
            CASCADE_SCALE_IMAGE,
            core::Size {
                width: 40,
                height: 40,
            },
            core::Size {
                width: 0,
                height: 0,
            },
        ) {
            Ok(v) => v,
            Err(_e) => panic!("no face_cascade"),
        }

        for face in faces {
            frame = put_moustache(frame, face.x, face.y, face.width, face.height);
        }

        highgui::imshow("Video", &frame).unwrap();

        if highgui::wait_key(1).unwrap() & 0xFF == 'q' as i32 {
            break;
        }
    }
    ()
}
