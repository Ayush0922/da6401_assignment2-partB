# Fine-Tuning GoogleNet for Image Classification

This project demonstrates how to fine-tune a pre-trained GoogleNet model for image classification using PyTorch. It includes functionalities for data loading, preprocessing, training, validation, testing, and visualization of results.

## Table of Contents

* [Installation](#installation)
    * [Linux](#linux)
    * [macOS](#macos)
    * [Windows](#windows)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Key Features](#key-features)
* [Dependencies](#dependencies)
* [Dataset](#dataset)
* [Fine-Tuning Strategies](#fine-tuning-strategies)
* [Optimizer Options](#optimizer-options)
* [Contributing](#contributing)
* [License](#license)

## Installation

Before running the project, you need to install Python and the required libraries. We recommend using a virtual environment to manage dependencies.

### Linux

1.  **Install Python 3 and pip:**
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Install virtualenv (optional but recommended):**
    ```bash
    pip3 install virtualenv
    ```

3.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    cd venv
    source bin/activate
    ```

4.  **Install the required Python libraries:**
    ```bash
    pip install torch torchvision torchaudio scikit-learn matplotlib seaborn
    ```
    If you have a CUDA-enabled GPU and want to use it for training, make sure you have the appropriate CUDA drivers and cuDNN installed. PyTorch will automatically detect and utilize the GPU if available. You might need to install a specific PyTorch version with CUDA support if the default installation doesn't work. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for CUDA-specific installation instructions.

### macOS

1.  **Install Python 3 and pip (if not already installed):**
    You can download the latest version of Python from the [official Python website](https://www.python.org/downloads/macos/) or use a package manager like Homebrew.

    **Using Homebrew:**
    ```bash
    brew update
    brew install python3
    ```

2.  **Install virtualenv (optional but recommended):**
    ```bash
    pip3 install virtualenv
    ```

3.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    cd venv
    source bin/activate
    ```

4.  **Install the required Python libraries:**
    ```bash
    pip install torch torchvision torchaudio scikit-learn matplotlib seaborn
    ```
    For GPU support on macOS (using Metal), ensure you have a compatible Apple silicon Mac and the latest version of macOS. PyTorch should automatically utilize the GPU if available.

### Windows

1.  **Install Python 3 and pip:**
    Download the latest version of Python from the [official Python website](https://www.python.org/downloads/windows/) and run the installer. Make sure to check the box that says "Add Python to PATH" during the installation.

2.  **Install virtualenv (optional but recommended):**
    Open Command Prompt or PowerShell and run:
    ```bash
    pip install virtualenv
    ```

3.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    cd venv
    .\Scripts\activate
    ```

4.  **Install the required Python libraries:**
    ```bash
    pip install torch torchvision torchaudio scikit-learn matplotlib seaborn
    ```
    If you have an NVIDIA GPU and want to use it for training, you need to install the appropriate NVIDIA drivers and CUDA toolkit. Then, install the PyTorch version that supports your CUDA version. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for CUDA-specific installation instructions.

## Project Structure

```
.
├── README.md
├── <b.py  # The main Python script provided
└── /kaggle/input/nature-922/       # (Assumed) Dataset directory
    └── inaturalist_12K/
        ├── train/
        │   ├── class_1/
        │   │   └── ...
        │   ├── class_2/
        │   │   └── ...
        │   └── ...
        └── val/
            ├── class_1/
            │   └── ...
            ├── class_2/
            │   └── ...
            └── ...
```

**Note:** The `/kaggle/input/nature-922/inaturalist_12K` directory structure is assumed based on the provided code. Ensure your dataset is organized similarly with `train` and `val` subdirectories, each containing subdirectories for different classes.

## Usage

To run the fine-tuning process, execute the main Python script. You can modify the parameters within the `fine_tune_model` function call to adjust the number of epochs, the number of layers to freeze (`k`), the freezing strategy (`strategy`), and the optimizer type (`optim_type`).

```bash
python <your_python_script_name>.py
```

For example, to train for 10 epochs, freeze the first 6 layers, use the "first" strategy, and the Adam optimizer, you would modify the script as follows:

```python
if __name__ == '__main__':
    trained_model = fine_tune_model(epoch=10, k=6, strategy="first", optim_type="adam")
    # You can add code here to save or further utilize the trained_model
```

## Key Features

* **Device Agnostic:** Automatically uses CUDA-enabled GPU if available, otherwise defaults to CPU.
* **Data Loading and Preprocessing:** Utilizes `torchvision.datasets.ImageFolder` for efficient loading of image data and applies necessary transformations, including resizing, normalization, and random horizontal flipping for training data augmentation.
* **Data Splitting:** Splits the training data into training and validation subsets using `torch.utils.data.random_split`.
* **Fine-Tuning:** Implements a function `fine_tune_model` that loads a pre-trained GoogleNet model and allows freezing of specific layers based on different strategies.
* **Layer Freezing Strategies:** Supports freezing the "first", "middle", or "last" `k` layers of the pre-trained model.
* **Optimizer Selection:** Offers options for using Adam or SGD optimizers.
* **Training and Validation Loop:** Includes a `forward_pass` function to handle both training and validation steps, calculating loss and accuracy.
* **Performance Monitoring:** Tracks and prints training and validation loss and accuracy per epoch.
* **Visualization:** Generates plots for training and validation accuracy curves and a normalized confusion matrix to visualize the classification performance on the test set.

## Dependencies

* **torch:** PyTorch deep learning framework.
* **torchvision:** Provides datasets, model architectures, and image transformations for PyTorch.
* **scikit-learn:** Used for calculating the confusion matrix.
* **matplotlib:** For plotting accuracy curves.
* **seaborn:** For creating the confusion matrix heatmap.
* **numpy:** For numerical operations.
* **pillow (PIL):** Image processing library (often a dependency of torchvision).

## Dataset

The code assumes the presence of an image dataset organized into class-specific subdirectories within `train` and `val` folders. The path to the root of this dataset is defined in the `prepare_dataloaders` function (`data_dir="/kaggle/input/nature-922/inaturalist_12K"` by default). You may need to adjust this path based on the location of your dataset.

## Fine-Tuning Strategies

The `freeze_layers` function allows you to freeze different sets of layers in the pre-trained GoogleNet model based on the `strategy` parameter:

* **"first"**: Freezes the first `k` layers of the model.
* **"middle"**: Freezes the middle `k` layers of the model.
* **"last"**: Freezes the last `k` layers of the model.

This allows you to experiment with different levels of feature adaptation during fine-tuning.

## Optimizer Options

The `fine_tune_model` function currently supports two optimizer types specified by the `optim_type` parameter:

* **"adam"**: Uses the Adam optimizer.
* **"sgd"**: Uses the Stochastic Gradient Descent (SGD) optimizer with a momentum of 0.9.

You can easily extend this dictionary in the `fine_tune_model` function to include other PyTorch optimizers.

## Contributing

Contributions to this project are welcome. Feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.

## License

This project is licensed under the [MIT License].
