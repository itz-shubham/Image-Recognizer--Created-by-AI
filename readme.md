

# AI-Powered Image Classification Web App: Built by AI  

## Overview  
This project showcases the power of AI in development. Using **tools like ChatGPT and Codium**, I created an image classification app without writing a single line of code myself! It uses the **ResNet50 model** to classify uploaded images and display results instantly on a web page. Even this README was written by AI!

## Requirements

* Python 3.10
* Flask
* Torch
* Torchvision
* Pillow
* NumPy
* Werkzeug

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

The project consists of the following files and directories:

* `app.py`: The main application file, containing the Flask app and routes.
* `templates/`: A directory containing the HTML templates for the application.
	+ `upload.html`: The template for the image upload page.
	+ `results.html`: The template for the classification results page.
* `requirements.txt`: A file listing the required dependencies for the project.

## Usage

To run the application, execute the following command:

```bash
python app.py
```

This will start the Flask development server, and the application will be available at `http://localhost:5000`.

## Image Upload and Classification

To classify an image, follow these steps:

1. Navigate to `http://localhost:5000` in your web browser.
2. Click the "Choose File" button to select an image file.
3. Click the "Analyze Image" button to upload and classify the image.
4. The classification results will be displayed on a new page.

## Classification Results

The classification results page displays the following information:

* The uploaded image.
* The top three predicted classes, along with their corresponding probabilities.
* A list of all predicted classes, along with their probabilities.

## Model and Dependencies

The application uses the pre-trained ResNet50 model, which is loaded using the `torchvision` library. The model is used to classify the uploaded images.

The application also uses the following dependencies:

* `Flask` for building the web application.
* `Torch` for loading and using the pre-trained model.
* `Torchvision` for loading and using the pre-trained model.
* `Pillow` for image processing.
* `NumPy` for numerical computations.
* `Werkzeug` for utility functions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project uses the pre-trained ResNet50 model, which was trained on the ImageNet dataset. The model is available under the Apache 2.0 license.

## Contributing

Contributions to this project are welcome. Please submit a pull request with your changes, and ensure that they align with the project's goals and coding standards.

## Issues

If you encounter any issues with the project, please submit a bug report or feature request using the GitHub issue tracker.