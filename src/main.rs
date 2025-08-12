mod data; // assuming you have data_reader.rs and forward_pass.rs in data/
use data::data_reader;
use data::forward_pass;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn cross_entropy_loss(predictions: &Vec<f32>, target: &Vec<f32>) -> f32 {
    let epsilon = 1e-15;
    let clipped_preds: Vec<f32> = predictions
        .iter()
        .map(|&p| p.clamp(epsilon, 1.0 - epsilon))
        .collect();

    target
        .iter()
        .zip(clipped_preds.iter())
        .map(|(&t, &p)| -t * p.ln())
        .sum()
}

fn main() {
    let pixel_x = 28;
    let pixel_y = 28;
    let number_hid = 10;
    let batch_size = 16; // mini-batch size
    let learning_rate = 0.01;
    let epochs = 20;

    let data_test = "data/mnist_test.csv";
    let data_train = "data/mnist_train.csv";

    let (test, train) = data_reader::Data_Checker(data_test, data_train);
    let (labels_test, images_test) = data_reader::read_mnist(test);
    let (labels_train, images_train) = data_reader::read_mnist(train);

    println!("Loaded {} test images", images_test.len());
    println!("Loaded {} training images", images_train.len());

    // Normalize images_train to 0..1 floats
    let norm_images_train: Vec<Vec<f32>> = images_train.iter()
        .map(|img| img.iter().map(|&p| p / 255.0).collect())
        .collect();

    let mut nn = forward_pass::NeuralNetwork::new(
        pixel_x * pixel_y,
        number_hid,
        10,
        norm_images_train.clone(),
        labels_train.clone(),
    );

    let mut rng = thread_rng();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct_count = 0;
        let mut total_samples = 0;

        let mut indices: Vec<usize> = (0..norm_images_train.len()).collect();
        indices.shuffle(&mut rng);

        for chunk in indices.chunks(batch_size) {
            for &idx in chunk {
                let image = &norm_images_train[idx];
                let target_label = &labels_train[idx];

                let hidden_output = nn.forward_hidden(image);
                nn.last_hidden_output = hidden_output.clone();

                let output_probs = nn.forward_output(&hidden_output);
                nn.last_output = output_probs.clone();

                //nn.backward(image, target_label, learning_rate);

                // Accumulate loss
                let loss = cross_entropy_loss(&output_probs, target_label);
                total_loss += loss;

                // Backpropagation
                nn.backward(image, target_label, learning_rate);

                // Accuracy check
                let predicted_class = output_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let target_class = target_label
                    .iter()
                    .position(|&v| v == 1.0)
                    .unwrap_or(usize::MAX);

                if predicted_class == target_class {
                    correct_count += 1;
                }

                total_samples += 1;
            }
        }

        let avg_loss = total_loss / total_samples as f32;
        let accuracy = correct_count as f32 / total_samples as f32 * 100.0;

        println!(
            "Epoch {}: Average cross-entropy loss: {:.6}, Accuracy: {:.2}%",
            epoch + 1,
            avg_loss,
            accuracy
        );
    }
}

