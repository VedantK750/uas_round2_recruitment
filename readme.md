## Introduction to Project
This project is a UAV image processing tool for search and rescue missions. It detects burnt and green grass regions, and identifies red and blue houses on both sides. This project uses **OpenCV** and **NumPy** to perform image segmentation and analysis. Additionally, the project calculates rescue priorities for houses based on their location.

## Features

- **Image Blurring:** Applies Gaussian blur for noise reduction.
- **HSV Conversion:** Converts the image to HSV color space for better segmentation.
- **Color Segmentation:** Detects burnt grass, green grass, red houses, and blue houses using color masks.
- **Contour Detection:** Finds and counts contours (houses) in specific regions.
- **Rescue Priority Calculation:** Determines rescue priorities for houses located on burnt and green grass areas.

## Libraries Used

- [OpenCV](https://opencv.org/): For image processing and computer vision tasks.
- [NumPy](https://numpy.org/): For handling numerical operations and arrays.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/VedantK750/uas_round2_recruitment/
    cd uas_round2_recruitment
    ```

2. Install required dependencies:
    ```bash
    pip install opencv-python numpy
    ```

## Usage

1. Place the image you want to analyze in the project folder and update the path in the code:
    ```python
    image = cv2.imread('/path/to/your/image.png')
    ```

2. Run the Python script:
    ```bash
    python uas_task_final.py
    ```

3. The output includes:
    - Segmented image showing burnt grass, green grass, red houses, and blue houses.
    - Number of houses on burnt and green regions.
    - Priority scores for each region.
    - A rescue ratio to guide priority of houses.

## Future Improvements

Some potential future improvements for this project include:
- **Enhanced Image Classification:** Using machine learning models to improve the accuracy of house and terrain classification.
- **Selecting better HSV values:** To create accurate masks

## Acknowledgments

Special thanks to the following resources and individuals:

- **OpenCV**: For providing an extensive and efficient library for image processing.
- **NumPy**: For simplifying matrix and array operations.
- **YouTube**: For guiding me on creating masks, creating contours and on how to count them.
- The **UAV team** for project inspiration and mission objectives.

## Contact

For any questions or suggestions, feel free to reach out:

- **Project Maintainer:** Vedant Shanker
- **Email:** [vedantkrish750@gmail.com](mailto:vedantkrish750@gmail.com)

---
Thank you