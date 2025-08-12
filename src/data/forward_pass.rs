use rand::Rng;

#[derive(Debug)]
pub(crate) struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<f32>,
    weights_hidden_output: Vec<f32>,
    biases_hidden: Vec<f32>,
    biases_output: Vec<f32>,
    pub(crate) neural_images: Vec<Vec<f32>>,
    pub(crate) neural_labels: Vec<Vec<f32>>,
    pub(crate) last_hidden_output: Vec<f32>,
    pub(crate) last_output: Vec<f32>,
}

impl NeuralNetwork {
    pub(crate) fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        images: Vec<Vec<f32>>,
        labels: Vec<Vec<f32>>,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden = (0..input_size * hidden_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let weights_hidden_output = (0..hidden_size * output_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Initialize biases to zero for better training stability
        let biases_hidden = vec![0.0; hidden_size];
        let biases_output = vec![0.0; output_size];

        let last_hidden_output = vec![0.0; hidden_size];
        let last_output = vec![0.0; output_size];

        Self {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            biases_hidden,
            biases_output,
            neural_images: images,
            neural_labels: labels,
            last_hidden_output,
            last_output,
        }
    }

    /// Softmax function (not currently used outside, but good to keep)
    #[allow(dead_code)]
    fn softmax(logits: &Vec<f32>) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max); // for numerical stability
        let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum_exp).collect()
    }

    pub(crate) fn forward_hidden(&self, input: &Vec<f32>) -> Vec<f32> {
    let mut result = vec![0.0; self.hidden_size];
    for j in 0..self.hidden_size {
        let mut sum = 0.0;
        for i in 0..self.input_size {
            sum += self.weights_input_hidden[j * self.input_size + i] * input[i];
        }
        sum += self.biases_hidden[j];
        result[j] = NeuralNetwork::relu(sum);
    }
    result
}

pub(crate) fn forward_output(&self, hidden: &Vec<f32>) -> Vec<f32> {
    let mut z = vec![0.0; self.output_size];
    for k in 0..self.output_size {
        let mut sum = 0.0;
        for j in 0..self.hidden_size {
            sum += self.weights_hidden_output[k * self.hidden_size + j] * hidden[j];
        }
        sum += self.biases_output[k];
        z[k] = sum;
    }
    NeuralNetwork::softmax(&z)
}

    // Backpropagation
    pub(crate) fn backward(&mut self, input: &Vec<f32>, target: &Vec<f32>, learning_rate: f32) {
        // 1. Calculate output error delta = y_pred - y_true
        let mut delta_output = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            delta_output[k] = self.last_output[k] - target[k];
        }

        // 2. Calculate hidden layer error delta_hidden = (W_hidden_output^T * delta_output) * relu'(hidden)
        let mut delta_hidden = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut sum = 0.0;
            for k in 0..self.output_size {
                sum += self.weights_hidden_output[k * self.hidden_size + j] * delta_output[k];
            }
            delta_hidden[j] = sum * relu_derivative(self.last_hidden_output[j]);
        }

        // 3. Update weights_hidden_output and biases_output
        for k in 0..self.output_size {
            for j in 0..self.hidden_size {
                let grad = delta_output[k] * self.last_hidden_output[j];
                self.weights_hidden_output[k * self.hidden_size + j] -= learning_rate * grad;
            }
            self.biases_output[k] -= learning_rate * delta_output[k];
        }

        // 4. Update weights_input_hidden and biases_hidden
        for j in 0..self.hidden_size {
            for i in 0..self.input_size {
                let grad = delta_hidden[j] * input[i];
                self.weights_input_hidden[j * self.input_size + i] -= learning_rate * grad;
            }
            self.biases_hidden[j] -= learning_rate * delta_hidden[j];
        }
    }

    // ReLU activation function
    fn relu(value: f32) -> f32 {
        if value < 0.0 {
            0.0
        } else {
            value
        }
    }
}

fn relu_derivative(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else {
        0.0
    }
}
