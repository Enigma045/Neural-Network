/*mod data;

use data::data_reader;
use std::fs::File;
use std::io::{BufRead,BufReader, Read, Write};
fn main() {
    let data_test = "data/mnist_test.csv";
    let data_train = "data/mnist_train.csv";
    
    let test = File::open(data_test);
    let train = File::open(data_train);
    
    let mut images: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();
    
    let mut pat_test = match test{
        Ok(test) => {
            test
        },
        Err(err) => {
            todo!()
        },
    };

    let mut pat_train = match train{
        Ok(train) => {
            train
        },
        Err(err) => {
            todo!()
        },
    };

    let mut content = String::new();
   
    pat_train.read_to_string(&mut content);

    let mut path = File::create("data.txt");

    let mut path = match path {
        Ok(path) => {
            println!("File created");
            path
        },
        Err(err) => {
            println!("{}",err);
            todo!()
        },
    };
    for (i,line) in content.lines().enumerate(){

        if i == 0 { continue; }

        let values: Vec<&str> = line.split(",").collect();

        //println!("First pixel: {}", values[0]);
        //Normalization
        let pixels: Vec<f32> = values[1..]
        .iter()
        .filter_map(|&p| {
        let trimmed = p.trim();
        match trimmed.parse::<f32>() {
            Ok(val) => Some(val / 255.0),
            Err(_) => {
                eprintln!("Skipping invalid pixel value: '{}'", trimmed);
                None
            }
        }
        })
        .collect();

        //one-hot
        let label = values[0].parse::<usize>().unwrap();
        let mut one_hot: [f32; 10] = [0.0;10];
        one_hot[label] = 1.0; 

        labels.push(one_hot.to_vec());
        images.push(pixels);
        
        //println!("{}",labels);
        //
        //path.write(line.as_bytes());
    }
    
    //&images[0][..783]

    println!("Loaded {} images", images.len());
    println!("First image label: {:?}", labels[0]);
    println!("First pixel values:");

    let mut i = 0;
    for x in 1..28{
        for y in 0..28{
        let jojo = format!("{},", &images[0][i].to_string());
        path.write(jojo.as_bytes());
        i += 1;
    }
    }
    //println!("First pixel values: {:?}", &images[0][..10]);
    
   //path.write(images[0][..786].split(",")); 
}
*/