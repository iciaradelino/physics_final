#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // For random initialization
#include <string.h> // For strtok, strcmp

// --- Configuration ---
#define INPUT_SIZE 32
#define HIDDEN_SIZE 4 // Starting small
#define OUTPUT_SIZE 2
#define NUM_EPOCHS 1000 // Example number of training iterations
#define LEARNING_RATE 0.1f // Example learning rate
#define NUM_TRAIN_SAMPLES 50 // Placeholder: Adjust when data is available
#define MAX_LINE_LENGTH 1024 // Max characters per CSV line

// --- Network Parameters (Static Allocation for Arduino) ---
float hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
float hidden_bias[HIDDEN_SIZE];
float output_weights[HIDDEN_SIZE][OUTPUT_SIZE];
float output_bias[OUTPUT_SIZE];

// Intermediate values needed for backpropagation
float hidden_weighted_sum[HIDDEN_SIZE]; // Store weighted sum before ReLU
float hidden_delta[HIDDEN_SIZE];        // Store error delta for hidden layer
float output_delta[OUTPUT_SIZE];       // Store error delta for output layer

// --- Training Data (Placeholders - To be filled by load_data) ---
float train_inputs[NUM_TRAIN_SAMPLES][INPUT_SIZE];
float train_targets[NUM_TRAIN_SAMPLES][OUTPUT_SIZE];
int actual_train_samples = 0; // Keep track of loaded samples

// --- Function Prototypes ---
void initialize_network();
void forward_pass(const float input[INPUT_SIZE], float hidden_output[HIDDEN_SIZE], float final_output[OUTPUT_SIZE]);
void backward_pass(const float input[INPUT_SIZE], const float target[OUTPUT_SIZE], const float hidden_output[HIDDEN_SIZE], const float final_output[OUTPUT_SIZE]);
void update_weights(const float input[INPUT_SIZE], const float hidden_output[HIDDEN_SIZE]);
float relu(float x);
float relu_derivative(float x);
void load_data(const char* filename); // Placeholder -> Implement
void train(); // Needs update to use actual_train_samples
void export_weights(const char* filename); // Placeholder

// --- Main Function ---
int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("Initializing network...
");
    initialize_network();

    // TODO: Load training data
    printf("Loading training data...\n");
    load_data("fft_diode_dataset.csv");

    // TODO: Implement training loop
    printf("Starting training...\n");
    train();

    // TODO: Export weights after training
    printf("Exporting weights...\n");
    export_weights("network_weights.txt");

    printf("Process finished.
");
    return 0;
}

// --- Function Implementations (To be filled in) ---

void initialize_network() {
    printf("  Initializing hidden weights and biases...
");
    // Initialize hidden layer weights and biases (small random values)
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            hidden_weights[i][j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f; // Small random values between -0.1 and 0.1
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        hidden_bias[j] = 0.0f; // Initialize biases to zero or small random values
    }

    printf("  Initializing output weights and biases...
");
    // Initialize output layer weights and biases
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            output_weights[i][j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        }
    }
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        output_bias[j] = 0.0f;
    }
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float relu_derivative(float x) {
    // Note: x here is the *input* to the relu function (the weighted sum + bias)
    // Derivative is 1 if x > 0, 0 otherwise
    return (x > 0.0f) ? 1.0f : 0.0f;
}

// --- Placeholder Functions ---

void forward_pass(const float input[INPUT_SIZE], float hidden_output[HIDDEN_SIZE], float final_output[OUTPUT_SIZE]) {
    // Calculate hidden layer activations (using ReLU)
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float weighted_sum = hidden_bias[j];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            weighted_sum += input[i] * hidden_weights[i][j];
        }
        // Store the weighted sum before activation for backpropagation
        hidden_weighted_sum[j] = weighted_sum;
        hidden_output[j] = relu(weighted_sum);
    }

    // Calculate output layer activations (using Linear)
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        float weighted_sum = output_bias[j];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            weighted_sum += hidden_output[i] * output_weights[i][j];
        }
        final_output[j] = weighted_sum; // Linear output layer
    }
}

void backward_pass(const float input[INPUT_SIZE], const float target[OUTPUT_SIZE], const float hidden_output[HIDDEN_SIZE], const float final_output[OUTPUT_SIZE]) {
    // Calculate output layer delta (Error * derivative_of_linear_activation)
    // Derivative of linear activation f(x)=x is 1.
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        output_delta[j] = final_output[j] - target[j]; // MSE derivative part
    }

    // Calculate hidden layer delta (using chain rule)
    // delta_hidden = (sum(delta_output * weight_output)) * derivative_of_hidden_activation
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float error_prop = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            error_prop += output_delta[k] * output_weights[j][k]; // Note index order for weights
        }
        hidden_delta[j] = error_prop * relu_derivative(hidden_weighted_sum[j]);
    }
}

void update_weights(const float input[INPUT_SIZE], const float hidden_output[HIDDEN_SIZE]) {
    // Update output layer weights and biases
    for (int j = 0; j < OUTPUT_SIZE; ++j) { // Iterate over output neurons
        for (int i = 0; i < HIDDEN_SIZE; ++i) { // Iterate over hidden neurons connections
            // Weight update = LearningRate * OutputDelta * HiddenActivation
            output_weights[i][j] -= LEARNING_RATE * output_delta[j] * hidden_output[i];
        }
        // Bias update = LearningRate * OutputDelta
        output_bias[j] -= LEARNING_RATE * output_delta[j];
    }

    // Update hidden layer weights and biases
    for (int j = 0; j < HIDDEN_SIZE; ++j) { // Iterate over hidden neurons
        for (int i = 0; i < INPUT_SIZE; ++i) { // Iterate over input neurons connections
            // Weight update = LearningRate * HiddenDelta * InputActivation
            hidden_weights[i][j] -= LEARNING_RATE * hidden_delta[j] * input[i];
        }
         // Bias update = LearningRate * HiddenDelta
        hidden_bias[j] -= LEARNING_RATE * hidden_delta[j];
    }
}

void load_data(const char* filename) {
    FILE *infile = fopen(filename, "r");
    if (infile == NULL) {
        perror("Error opening dataset file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;

    // Skip header line
    if (fgets(line, sizeof(line), infile) == NULL) {
        fprintf(stderr, "Error reading header or empty file.\n");
        fclose(infile);
        return;
    }

    // Read data lines
    while (fgets(line, sizeof(line), infile) != NULL && sample_count < NUM_TRAIN_SAMPLES) {
        char *token;
        int feature_index = 0;

        // Get features (first 32 columns)
        token = strtok(line, ",");
        while (token != NULL && feature_index < INPUT_SIZE) {
            train_inputs[sample_count][feature_index++] = atof(token);
            token = strtok(NULL, ",");
        }

        // Get the label (last column)
        if (token != NULL) {
            // Remove trailing newline if present
            size_t len = strlen(token);
            if (len > 0 && token[len - 1] == '\n') {
                token[len - 1] = '\0';
            }
            // Remove trailing carriage return if present (for Windows CRLF)
             if (len > 1 && token[len - 2] == '\r') {
                token[len - 2] = '\0';
            }


            if (strcmp(token, "forward_biased_diode") == 0) {
                train_targets[sample_count][0] = 1.0f;
                train_targets[sample_count][1] = 0.0f;
            } else if (strcmp(token, "reverse_biased_diode") == 0) {
                train_targets[sample_count][0] = 0.0f;
                train_targets[sample_count][1] = 1.0f;
            } else {
                fprintf(stderr, "Warning: Unknown label '%s' on line %d\n", token, sample_count + 2); // +2 for 1-based index and header
                // Optionally skip this sample or assign a default target
                 continue; // Skip this sample if label is unknown
            }
             sample_count++;
        } else {
             fprintf(stderr, "Warning: Incomplete data on line %d\n", sample_count + 2);
        }

    }

    fclose(infile);
    actual_train_samples = sample_count;
    printf("Loaded %d training samples from %s\n", actual_train_samples, filename);

     if (actual_train_samples == 0) {
        fprintf(stderr, "Error: No valid training samples loaded.\n");
    } else if (actual_train_samples < NUM_TRAIN_SAMPLES) {
        printf("Warning: Dataset contains fewer samples (%d) than defined NUM_TRAIN_SAMPLES (%d).\n", actual_train_samples, NUM_TRAIN_SAMPLES);
    }
}

void train() {
    if (actual_train_samples == 0) {
        fprintf(stderr, "Error: Cannot train, no data loaded.\n");
        return;
    }
    printf("Starting training for %d epochs on %d samples...\n", NUM_EPOCHS, actual_train_samples);

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float total_epoch_error = 0.0f;

        // Loop through each training sample
        for (int sample_idx = 0; sample_idx < actual_train_samples; ++sample_idx) {
            // Temporary arrays for outputs for this sample
            float hidden_output[HIDDEN_SIZE];
            float final_output[OUTPUT_SIZE];

            // 1. Forward Pass
            forward_pass(train_inputs[sample_idx], hidden_output, final_output);

            // 2. Calculate Error (MSE for this sample)
            float sample_error = 0.0f;
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                float error = final_output[i] - train_targets[sample_idx][i];
                sample_error += error * error;
            }
            total_epoch_error += sample_error / OUTPUT_SIZE;

            // 3. Backward Pass (Calculate deltas)
            backward_pass(train_inputs[sample_idx], train_targets[sample_idx], hidden_output, final_output);

            // 4. Update Weights
            update_weights(train_inputs[sample_idx], hidden_output);
        }

        // Print epoch status (e.g., every 100 epochs)
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            printf("Epoch %d/%d, Average Error: %f\n",
                   epoch + 1, NUM_EPOCHS, total_epoch_error / actual_train_samples);
        }
    }
    printf("Training finished.\n");
}

void export_weights(const char* filename) {
    FILE *outfile = fopen(filename, "w");
    if (outfile == NULL) {
        perror("Error opening file for writing weights");
        return;
    }

    fprintf(outfile, "// Neural Network Weights and Biases\n");
    fprintf(outfile, "// Input Size: %d, Hidden Size: %d, Output Size: %d\n\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // --- Hidden Layer Weights ---
    fprintf(outfile, "hidden_weights[%d][%d]:\n", INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            fprintf(outfile, "%f ", hidden_weights[i][j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

    // --- Hidden Layer Biases ---
    fprintf(outfile, "hidden_bias[%d]:\n", HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        fprintf(outfile, "%f ", hidden_bias[j]);
    }
    fprintf(outfile, "\n\n");

    // --- Output Layer Weights ---
    fprintf(outfile, "output_weights[%d][%d]:\n", HIDDEN_SIZE, OUTPUT_SIZE);
     for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            fprintf(outfile, "%f ", output_weights[i][j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

    // --- Output Layer Biases ---
    fprintf(outfile, "output_bias[%d]:\n", OUTPUT_SIZE);
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        fprintf(outfile, "%f ", output_bias[j]);
    }
    fprintf(outfile, "\n");

    fclose(outfile);
    printf("Weights exported successfully to %s\n", filename);
}
