# 🎮 Kinderuniversity: Learn Computer Vision Through Play

[![Made for Education](https://img.shields.io/badge/Educational%20Purpose-✓-brightgreen)](https://github.com)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-brightgreen)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Cv11-informational)](https://github.com/ultralytics/ultralytics)

## 🚀 Introduction

**Kinderuniversity** is an interactive educational platform designed to introduce Computer Vision (CV) concepts in a fun, engaging, and hands-on way. Whether you're a teacher looking for an easy entry point to explain CV to students, or an educator wanting to challenge learners through a game-based approach, this project provides everything you need.

Transform your classroom into a CV exploration lab where students don't just *learn about* computer vision—they *experience* it through:
- **Conceptual presentations** explaining CV fundamentals
- **Interactive visualizations** of real-time image processing pipelines
- **A treasure hunt game** powered by object detection and pose estimation

---

## 📚 Three-Part Learning Journey

### 1️⃣ **Presentation** — *Understand the Concepts*
Start with fundamentals. Clear explanations of:
- What is Computer Vision?
- How do machines "see"?
- Introduction to neural networks and deep learning
- Real-world applications

👉 **Location:** `1_Presentation/`

---

### 2️⃣ **Practical Exercise** — *Visualize the Pipeline*
Get hands-on with actual CV algorithms. Run Jupyter notebooks that visualize:
- **Image Processing Filters** (Edge detection, color spaces, transformations)
- **Real-time Computer Vision Pipeline** using your webcam
- **Deep Learning Detection** with YOLO11 and YOLO8
- **Interactive sliders** to adjust algorithm parameters and see instant results

**Perfect for:**
- Understanding how different CV filters work
- Experimenting with hyperparameters
- Building intuition about image processing

👉 **Location:** `2_Practical Exercise/`
- `251024_Final_VisualizingComputerVisionPipeline.ipynb` — The main interactive notebook

---

### 3️⃣ **Game** — *Master CV Through Challenge*
A **treasure hunt game** that brings it all together! 

Players navigate through 7 progressive levels, each with unique challenges:
- **Level 1-3:** Object detection challenges (find specific items in real-time)
- **Level 4-7:** Pose estimation challenges (mimic poses, perform rituals)

🎯 **Game Features:**
- Multi-language support (English, Dutch, German, Spanish, French)
- Progressive difficulty
- Real-time CV processing
- Engaging visuals and sound effects
- Dynamic object detection with YOLO
- Pose estimation with skeletal tracking

👉 **Location:** `3_Game/`
- `251025_CVgame_treasure-hunt.ipynb` — Launch the game here

---

## 🎯 Who Is This For?

### 👨‍🏫 **For Teachers**
Looking for a structured way to introduce Computer Vision to motivated students? Kinderuniversity provides:
- ✅ Low setup complexity — just install Python packages
- ✅ Progressive difficulty — start simple, get complex
- ✅ Engaging content — move from theory to practice to game
- ✅ Extensible framework — easily add new levels and challenges

### 👨‍💻 **For Students**
Ready for a hands-on challenge? You'll:
- ✅ Learn CV fundamentals through interactive experiments
- ✅ See real algorithms working on your own webcam feed
- ✅ Build problem-solving skills through game-based learning
- ✅ Customize the game with your own challenges and levels

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.8 or later**
- **Webcam** (required for practical exercises and game)
- **GPU recommended** (NVIDIA CUDA for faster processing) — CPU works but slower

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Kinderuniversity.git
cd Kinderuniversity
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages:**
- `jupyter` — Interactive notebooks
- `opencv-python` — Computer vision library
- `numpy`, `torch` — Numerical computing
- `ultralytics` — YOLO object detection
- `pygame` — Game engine

### 3. Download Pre-trained Models
The game and exercises use pre-trained YOLO models. They'll auto-download on first run, or manually download:
- `yolo11x-pose.pt` → `3_Game/yolo-Weights/`
- `yolov8x-worldv2.pt` → `3_Game/yolo-Weights/`

### 4. Start Learning!

**Option A: Step through fundamentals**
```bash
cd 1_Presentation
# Open the presentation slides (pptx files)
```

**Option B: Interactive visualization**
```bash
cd 2_Practical\ Exercise
jupyter notebook 251024_Final_VisualizingComputerVisionPipeline.ipynb
```

**Option C: Play the game**
```bash
cd 3_Game
jupyter notebook 251025_CVgame_treasure-hunt.ipynb
# Run the first cell and enjoy!
```

---

## 🎮 The Treasure Hunt Game — In Action

The game is the centerpiece of this project. Players navigate through themed levels using **real-time computer vision**:

- **Object Detection Levels:** Find hidden treasures by detecting specific objects in your environment
- **Pose Estimation Levels:** Strike poses or perform ritual gestures to unlock challenges

**Supported Languages:**
- 🇬🇧 English
- 🇳🇱 Dutch
- 🇩🇪 German
- 🇪🇸 Spanish
- 🇫🇷 French

**Features:**
- Multi-level progression
- Dynamic leaderboard (track scores locally)
- Sound effects and speech bubbles
- Customizable player avatars
- Map-based navigation between levels

---

## 📁 Project Structure

```
Kinderuniversity/
├── 1_Presentation/               # Theory & concepts
│   ├── Kinderuniversiteit Presentation - EN.pptx
│   ├── Kinderuniversiteit Presentation - NL.pptx
│   └── Kinderuniversiteit Presentation - NL.pdf
│
├── 2_Practical Exercise/         # Hands-on learning
│   ├── 251024_Final_VisualizingComputerVisionPipeline.ipynb
│   └── templates/
│
├── 3_Game/                       # Treasure hunt game
│   ├── 251025_CVgame_treasure-hunt.ipynb  (Start here!)
│   ├── CV_navigation.py
│   ├── CV_pose_estimation.py
│   ├── game_runtime_utils.py
│   ├── maps/                     # Game levels
│   ├── players/                  # Character assets
│   ├── sounds/                   # Audio effects
│   ├── yolo-Weights/             # Pre-trained models
│   └── dialogs_*.json            # Multi-language support
│
└── players/                      # Player character resources
```

---

## 🛠️ How the Game Works (Technical Overview)

### Detection Pipeline
1. **Webcam Input** → Raw video frames
2. **YOLO Processing** → Object/pose detection
3. **Game Logic** → Check if detection matches challenge
4. **Feedback** → Visual and audio feedback to player

### Two Challenge Types

**Object Detection (Levels 1-3):**
- Find: Hat, Scissors, Lighter, Spoon, Necklace, Frisbee, Branch, Book
- Algorithm: YOLO v8 World (zero-shot detection)

**Pose Estimation (Levels 4-7):**
- Strike specific poses to unlock challenges
- Algorithm: YOLO11 Pose (skeleton keypoint detection)
- Example: Touch your toes, raise your arms, perform a ritual

---

## 🎨 Customization & Extension

### Add New Levels
New game levels can be added by:
1. Creating a new folder in `3_Game/maps/level_N/`
2. Defining game rules in the main game notebook
3. Adding objects to detect or poses to match

### Add New Objects to Detect
Edit the object list in `CV_navigation.py`:
```python
self.class_universe = [
    'hat', 'scissors', 'lighter', 
    'your_new_object_here'  # → Add here!
]
```

### Translate to New Languages
Add a new JSON file: `dialogs_YOUR_LANGUAGE.json` with translations

---

## 🔧 Troubleshooting

### Webcam Not Working?
- Check permissions: Is the camera allowed in your OS settings?
- Try: `python -c "import cv2; print(cv2.getBuildInformation())"`

### Slow Performance?
- Enable GPU: `torch.cuda.is_available()` should return `True`
- Reduce video resolution in game settings
- Close other applications

### Models Not Downloading?
- Check internet connection
- Manually download from [Ultralytics releases](https://github.com/ultralytics/ultralytics)
- Place `.pt` files in `3_Game/yolo-Weights/`

### Import Errors?
- Ensure all packages installed: `pip install -r requirements.txt`
- Restart your Jupyter kernel
- Update packages: `pip install --upgrade -r requirements.txt`

---

## 🎓 Learning Outcomes

By completing this project, learners will:

✅ **Understand how machines see** — Image sensors, pixels, color spaces
✅ **Grasp deep learning basics** — Neural networks, feature extraction, inference
✅ **Apply CV in practice** — Real-time object detection and pose estimation
✅ **Problem-solve creatively** — Design challenges within the game framework
✅ **Think algorithmically** — Understand trade-offs (speed vs. accuracy)

---

## 🤝 Contributing

Have ideas for new challenges, levels, or features? We'd love your contributions!

### Ways to Help:
1. **Add new game levels** with unique CV challenges
2. **Translate to more languages**
3. **Improve games assets** (characters, sounds, visuals)
4. **Bug fixes and optimizations**
5. **Documentation improvements**

Please submit:
- Issues for bugs or suggestions
- Pull requests for improvements
- Feature requests for new CV challenges

---

## 📋 Requirements

- **Python:** 3.8 or later
- **RAM:** 4GB minimum (8GB+ recommended)
- **GPU:** Optional but recommended (NVIDIA CUDA)
- **Webcam:** Required for exercises and game
- **Display:** 1920x1080 or higher recommended

### Python Packages:
See `requirements.txt` for full dependency list. Key packages:
- torch / torchvision
- opencv-python
- ultralytics (YOLO)
- jupyter / jupyterlab
- pygame
- numpy

---

## 📜 License

This project is provided for educational purposes. See LICENSE file for details.

---

## 👥 Credits & Acknowledgments

Created for **Kinderuniversity** — an initiative to bring Computer Vision education to the next generation.

Special thanks to:
- [Ultralytics](https://www.ultralytics.com/) for YOLO models
- [OpenCV](https://opencv.org/) for computer vision tools
- [Pygame](https://www.pygame.org/) for the game engine
- All educators and students who inspired this project

---

## 📞 Support

Getting stuck? Have questions?

- 📧 **Open an issue** on GitHub
- 💬 **Start a discussion** for general questions
- 📚 **Check the wikis** for detailed guides

---

## 🌟 Next Steps

1. **Try the interactive notebook** → `2_Practical Exercise/`
2. **Explore the code** → Understand how detection works
3. **Play the game** → `3_Game/251025_CVgame_treasure-hunt.ipynb`
4. **Design your own challenge** → Add a new level!
5. **Share your success** → Show us what you built!

---

**Happy learning! 🚀 Let's teach the world to see through machine learning.** 🤖👀

---

*Kinderuniversity makes Computer Vision accessible, engaging, and fun.*
