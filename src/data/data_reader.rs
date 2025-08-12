use std::{fs::File, io::{Read, Write}};



pub(crate) fn Data_Checker(data_test: &'static str,data_train: &'static str)->(File,File){
    
    let test = File::open(data_test);
    let train = File::open(data_train);
    
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

    (pat_test,pat_train)
}

fn One_hot_encording(values:Vec<&str>)->[f32;10] {
let label = values[0].parse::<usize>().unwrap();
        let mut one_hot: [f32; 10] = [0.0;10];
        one_hot[label] = 1.0; 

    one_hot
}

fn Normalization(values:Vec<&str>)-> Vec<f32>{
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

    pixels
}

fn Create_File(images:Vec<Vec<f32>>){

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
    
    let mut i = 0;
    for x in 1..28{
        for y in 0..28{
        let jojo = format!("{},", &images[0][i].to_string());
        path.write(jojo.as_bytes());
        i += 1;
    }
    }
}

pub(crate) fn read_mnist(mut file:File)-> (Vec<Vec<f32>>, Vec<Vec<f32>>){

    let mut images: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();
    
    let mut content = String::new();
   
    file.read_to_string(&mut content);

    for (i,line) in content.lines().enumerate(){

        if i == 0 { continue; }

        let values: Vec<&str> = line.split(",").collect();


        labels.push(One_hot_encording(values.clone()).to_vec());
        images.push(Normalization(values.clone()));    
    }

    (labels,images)
}