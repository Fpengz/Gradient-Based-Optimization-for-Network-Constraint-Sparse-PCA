import numpy as np

def get_pitprop_correlation_matrix():
    """
    Returns the Pitprop correlation matrix (13x13).
    Source: Jeffers (1967).
    """
    # Lower triangle of the correlation matrix
    lower = [
        [1.000],
        [0.954, 1.000],
        [0.386, 0.322, 1.000],
        [0.322, 0.268, 0.984, 1.000],
        [0.304, 0.246, 0.952, 0.981, 1.000],
        [0.340, 0.283, 0.735, 0.696, 0.696, 1.000],
        [0.287, 0.239, 0.576, 0.527, 0.522, 0.903, 1.000],
        [0.179, 0.166, 0.045, 0.039, 0.050, 0.408, 0.457, 1.000],
        [0.222, 0.198, 0.138, 0.113, 0.149, 0.456, 0.472, 0.912, 1.000],
        [0.157, 0.117, 0.115, 0.103, 0.107, 0.352, 0.357, 0.590, 0.601, 1.000],
        [-0.128, -0.114, -0.055, -0.050, -0.056, -0.177, -0.157, -0.257, -0.280, -0.067, 1.000],
        [-0.126, -0.107, -0.050, -0.048, -0.054, -0.141, -0.121, -0.185, -0.198, -0.059, 0.974, 1.000],
        [-0.054, -0.048, -0.013, -0.014, -0.014, -0.055, -0.054, -0.076, -0.084, -0.019, 0.469, 0.442, 1.000]
    ]
    
    matrix = np.zeros((13, 13))
    for i in range(13):
        for j in range(len(lower[i])):
            matrix[i, j] = lower[i][j]
            matrix[j, i] = lower[i][j]
            
    return matrix

if __name__ == "__main__":
    corr = get_pitprop_correlation_matrix()
    print("Pitprop correlation matrix shape:", corr.shape)
    print("First 3 rows/cols:")
    print(corr[:3, :3])
