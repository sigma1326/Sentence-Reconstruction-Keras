

# Sentence Reconstruction

The purpose of this project is to take in input a sequence of words corresponding to a random permutation of a given English sentence and reconstruct the original sentence.
The output can be either produced in a single shot or through an iterative (autoregressive) loop generating a single token at a time.

CONSTRAINTS:

* No pretrained model can be used.
* The neural network models should have less the 20M parameters.
* No postprocessing should be done (e.g. no beam search)
* Any additional training data cannot be used.

This script contains functions and configurations used for training and evaluating various transformer models on sequence data. The main focus of this analysis is to optimize model performance, efficiency, and accuracy by experimenting with different configurations such as the number of encoder and decoder layers, embedding dimension, attention heads, intermediate dimension, dense layers, and loss functions.

## Examples

Below are examples demonstrating the usage of the primary functions within this script:

1. **Calculate Score**: You can calculate the similarity score between two sentences using the `score` function as shown in the example below:

```python
from difflib import SequenceMatcher

# Define two sample sentences
sentence1 = "This is a test sentence."
sentence2 = "This is another test sentence."

# Calculate and print the similarity score
similarity_score = SequenceMatcher(None, sentence1, sentence2).ratio()
print("Similarity Score:", similarity_score)
```

2. **Run Model on Scrambled Inputs**: To generate output sequences for scrambled input sequences using a trained model, you can use the `run_model_on_scrambled_inputs` function as follows:

```python
import numpy as np
from tensorflow import keras

# Assume a pre-trained model
model = keras.models.Sequential()

# Define input sequences
input_sequences = np.array([[1, 2, 3, 4, 0], [5, 6, 7, 2, 0]])

# Generate and print output sequences
output_sequences = run_model_on_scrambled_inputs(model, input_sequences)
print("Output Sequences:", output_sequences)
```

3. **Calculate Scores for Test Sequences**: To calculate scores for test sequences using a model or base input data, you can use the `calculate_score_for_test_sequences` function as shown in the example below:

```python
import collections

# Calculate and print the base score for the test data without the model
base_scores = calculate_score_for_test_sequences()

# Calculate and print the score for the test data with the model
model_scores = calculate_score_for_test_sequences(num_samples=4000, use_model=True)
```

## Contents

- **Score Calculation**: Function to calculate the similarity score between two strings using the SequenceMatcher algorithm.
- **Model Inference**: Function to run a model on scrambled inputs and return the predicted output sequences.
- **Special Tokens Removal**: Function to remove special tokens "<start>" and "<end>" from a sentence.
- **Score Calculation for Test Sequences**: Function to calculate scores for test sequences using either a model or base input data.

## Key Findings

The baseline configuration used 6 encoder and decoder layers resulting in approximately 14 million parameters and achieved a score of 0.51 on the training set.

Different configurations were tested to analyze their impact on model performance, efficiency, and accuracy:

- **Model Parameter Analysis**: Increasing complexity may not necessarily lead to improved accuracy in all cases. Modifying the number of encoder and decoder layers from 6 to 10 increased model parameters to approximately 14 million but yielded similar performance.
- **Attention Heads Analysis**: Changing the number of attention heads to 20 resulted in a lower score (0.50) compared to the 20M model and started overfitting after the 14th epoch, highlighting the importance of careful tuning.
- **Intermediate Dimension Analysis**: Modifying the intermediate dimension did not significantly impact performance, resulting in a model with approximately 14 million parameters similar to other configurations.
- **Dense Layers and Loss Function Analysis**: Adding additional dense layers before the main dense layer of the neural network resulted in a model with 11.5 million parameters but did not improve performance significantly. Using sparse categorical crossentropy instead of categorical crossentropy led to improved training time, reduced RAM usage, and comparable performance.
- **Final Configuration**: For the final configuration, an embedding size of 128 and layer normalization before the final dense layer were used. This resulted in a model with approximately 11 million parameters, which performed similarly to the 20M model, demonstrating that optimal results can be achieved with careful configuration tuning and the right balance between complexity and efficiency.

Overall, these experiments demonstrate the importance of careful tuning and the right balance between model complexity and efficiency in achieving competitive results.
