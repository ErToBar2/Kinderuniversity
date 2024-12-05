"How Machines See and Learn"
This project is a collection of educational materials developed for a workshop hosted at Kinderuniversiteit in Brugge on October 26, 2024.

The goal of this workshop was to introduce children aged 8–11 to the fascinating world of computer vision and machine learning through engaging, hands-on activities. Now, these materials are available for anyone to use, adapt, improve, add levels, share and enjoy!
In the python based game, the players are navigated only based on live video feed input.

🧰 Repository Contents:

1. Theory (1_Theory)
A PowerPoint presentation explaining the fundamentals of:
- Digital images
- RGB color spaces
- Filters (convolutions)
- Deep learning and convolutional neural networks (CNNs)
- Visual aids to demonstrate how machines differentiate between objects.

2. Practical Exercise (2_Practical_Exercise)
A Python notebook (notebook.ipynb) demonstrating:
- How machines perceive digital images (RGB visualization)
- Convolutions and edge detection (e.g., Canny)
- Object detection techniques, from traditional to modern.
- A live video feed example showcasing these concepts interactively.

3. Treasure Hunt Game (game.ipynb)
A custom Python game where players navigate using computer vision techniques.
Features include:
- Human pose estimation for character movement and interactions.
- YOLOv8x-world object detection to identify and use items in the game.
- A storytelling intro video (Intro.zip) to set the context of the adventure.

🚀 Who Is This For?
This repository is perfect for:
Educators introducing STEM concepts to young learners.
Schools, universities, and organizations hosting workshops on AI and computer vision.
Curious learners of all ages interested in understanding the science behind AI.

📜 Explore Each Folder:
Start with the PowerPoint presentation in 1_Theory.
Dive into interactive examples with the Python notebook in 2_Practical_Exercise.
Play the Treasure Hunt game using game.ipynb.

🚀 Environment Setup Instructions

Step 1: Create a New Conda Environment in Anaconda prompt (copy paste the following)
conda create -n pygame_yolo python=3.10
conda activate pygame_yolo
pip install ultralytics
pip install pygame

Step 2: Install PyTorch with CUDA Support (ultralytics commonly installs CPU verison of PyTorch)
nvcc --version

pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ### Replace cu121 with your specific CUDA version.### 

Step 3: Run code multiple times
Ultralytics will download all models and upates needed for the code. check printouts and restart and run the code a few times. 


🎓 Educational Goals
This workshop is designed to:
Simplify complex AI concepts for young minds.
Demonstrate how machines process visual data.
Inspire curiosity and creativity in exploring AI and machine learning.

🙌 Contributing and Feedback
Feel free to fork this repository and adapt the materials for your own workshops or educational sessions. Contributions are welcome! If you have any questions, suggestions, or feedback, please reach out via email.

📜 License
This project is open-source and available under the MIT License. Use it, share it, and help inspire the next generation of innovators.

Happy learning!
Erkki Bartczak
