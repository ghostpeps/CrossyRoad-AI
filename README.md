# Crossy Road AI (Deep Q-Network)

An autonomous AI agent trained to play and beat Crossy Road using **Reinforcement Learning (DQN)** and **Computer Vision**. 

This script is fully optimized for **macOS** and automatically handles screen monitoring and input injections without requiring any external template image files!

---

## Requirements & Installation

To run this AI on your own machine, you need a Mac with Python 3 installed. Follow these steps to set up the dependencies:

### 1. Clone the Repository
Open your Terminal and clone this project folder to your computer:
```bash
git clone https://github.com
cd CrossyRoad-AI
```
*(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username!)*

### 2. Install Python Dependencies
Run the following command to install the required computer vision, window automation, and machine learning libraries:
```bash
pip3 install torch torchvision mss opencv-python pyautogui --break-system-packages
```

---

## Mandatory macOS Security Setup

Because this Python script automatically monitors your screen layout and sends arrow keys to your browser, macOS will block it by default. You **must** grant permissions manually:

1. Open your Mac's **System Settings > Privacy & Security > Accessibility**.
2. Click the `+` icon and add your **Terminal** application (or **Visual Studio Code** if running directly from an editor). Turn the toggle **ON**.
3. Go to **Privacy & Security > Screen Recording**. 
4. Add your **Terminal** or **Visual Studio Code** here as well and turn the toggle **ON**.

---

## How to get to Crossy Road

1. Go to https://crossyroadgame.io/
2. Enable the fullscreen option
4. If the screen goes black, just click on the screen
5. Click play now

## How to Run the AI

1. Launch the AI script by running this command in your Terminal:
   ```bash
   python3 crossy_road_ai.py
   ```
2. **Take your hands off your keyboard and mouse!** The script will automatically pull Google Chrome to the front and begin training the chicken.

**If the program stops at the Game Over phase, simply click the enter/return key and the AI will take over.**

### How to Pause / Stop Training
Simply click away from Google Chrome onto your desktop background or another application. The script's built-in safety monitor will instantly freeze the neural network tensors, save the current model weights to a file named `crossy_road_brain.pth`, and shut down safely.
