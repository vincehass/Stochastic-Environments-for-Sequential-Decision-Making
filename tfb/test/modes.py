import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_num_modes(sequences, distance_threshold=1):
    # Ensure sequences is a 2D numpy array
    sequences = np.array(sequences)
    if sequences.ndim == 1:
        sequences = sequences.reshape(-1, 1)
    
    # If sequences are strings, convert to integer representation
    if sequences.dtype.kind in ['U', 'S']:  # Unicode or byte string
        unique_chars = np.unique(sequences.ravel())
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_sequences = np.array([[char_to_int[char] for char in seq] for seq in sequences])
    else:
        int_sequences = sequences
    
    # Ensure int_sequences is 2D
    if int_sequences.ndim == 1:
        int_sequences = int_sequences.reshape(-1, 1)
    
    # Calculate pairwise Hamming distances
    distances = squareform(pdist(int_sequences, metric='hamming'))
    print("Pairwise distances:\n", distances)
    
    # Initialize modes
    modes = []
    for i in range(len(sequences)):
        # Check if the current sequence is a mode
        is_mode = True
        for j in modes:
            # Adjust the condition to allow for more flexibility
            if distances[i, j] < distance_threshold / len(sequences[0]):
                is_mode = False
                break
        if is_mode:
            modes.append(i)
    
    # Return the unique modes based on their indices
    unique_modes = np.unique(modes)
    return len(unique_modes)

# Test the function
if __name__ == "__main__":
    # Define a set of sequences for testing
    sequences = [
        "abc",
        "abd",
        "xyz",
        "xzy",
        "abc",
        "abz"
    ]

    # Set a distance threshold
    distance_threshold = 1  # You can experiment with this value

    # Calculate the number of modes
    num_modes = calculate_num_modes(sequences, distance_threshold)
    print(f"Number of modes: {num_modes}")
