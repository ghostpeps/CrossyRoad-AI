import os
import random
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from mss import mss

# =====================================================================
# 1. SETUP, MACROS, & HYPERPARAMETERS
# =====================================================================
device = torch.device(
    'mps'
    if torch.backends.mps.is_available()
    else ('cuda' if torch.cuda.is_available() else 'cpu')
)

BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MEMORY_SIZE = 5000
TARGET_UPDATE = 10

pyautogui.PAUSE = 0.001  # ⚡ Maximize keyboard speed
ACTIONS = {0: 'up', 1: 'left', 2: 'right', 3: 'down'}

# Initialize the global fast screen grabber
sct = mss()


# =====================================================================
# 2. AUTOMATED DESKTOP ORCHESTRATION
# =====================================================================
def bring_browser_to_front():
    """Brings Chrome to the front layout but leaves window size unchanged."""
    print("Automating desktop windows... Bringing Google Chrome forward")
    subprocess.run(
        ["osascript", "-e", 'tell application "Google Chrome" to activate'],
        check=False,
    )
    time.sleep(1.0) 


# =====================================================================
# 3. ENVIRONMENT VISION & COLOR-BASED DEATH DETECTION
# =====================================================================
def get_ai_vision_state():
    """Ultra-fast screen capture using mss hardware-level calls."""
    monitor = sct.monitors[1]  # Capture primary display coordinates
    screenshot = sct.grab(monitor)
    screen = np.array(screenshot)

    gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (84, 84))

    state = resized.astype(np.float32) / 255.0
    return np.expand_dims(state, axis=0)


def check_if_dead():
    """
    Smarter Universal Death Check: Scans the center of the screen for the 
    distinct bright orange-red 'FREE GIFT' / Game Over banner.
    """
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    raw_screen = np.array(screenshot)

    # Convert from BGRA to standard RGB color format
    rgb_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2RGB)
    
    height, width, _ = rgb_screen.shape
    
    # Crop to a box right in the middle-top of your monitor screen
    # This targets the exact layout zone where the FREE GIFT popup spawns
    top = int(height * 0.15)
    bottom = int(height * 0.55)
    left = int(width * 0.25)
    right = int(width * 0.75)
    banner_zone = rgb_screen[top:bottom, left:right]
    
    # Crossy Road's signature bright orange-red banner hex/RGB profile boundaries
    lower_orange = np.array([220, 70, 20])   
    upper_orange = np.array([255, 120, 60])  
    
    # Create an isolation mask that lights up white only on matching orange pixels
    mask = cv2.inRange(banner_zone, lower_orange, upper_orange)
    orange_pixel_count = np.sum(mask > 0)
    
    # If a large cluster of orange pixels spawns, the game over banner is present!
    if orange_pixel_count > 3000:
        return True
    return False


# =====================================================================
# 4. THE AI'S BRAIN (CONVOLUTIONAL NEURAL NETWORK)
# =====================================================================
class CrossyRoadBrain(nn.Module):

    def __init__(self, action_space_size=3):
        super(CrossyRoadBrain, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =====================================================================
# 5. DATA MANAGEMENT (REPLAY EXPERIENCE BUFFER)
# =====================================================================
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# =====================================================================
# 6. CORE REINFORCEMENT MAIN TRAINING LOOP
# =====================================================================
def main():
    policy_net = CrossyRoadBrain(action_space_size=len(ACTIONS)).to(device)
    target_net = CrossyRoadBrain(action_space_size=len(ACTIONS)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    if os.path.exists('crossy_road_brain.pth'):
        print("Loading previously saved brain progress...")
        policy_net.load_state_dict(torch.load('crossy_road_brain.pth'))
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    bring_browser_to_front()

    print(f"Brain initialized: {device}")
    print("Commencing AI Training iterations...")

    for episode in range(1, 1001):
        state = get_ai_vision_state()
        total_reward = 0
        done = False

        while not done:
            if random.random() <= epsilon:
                action = random.randint(0, len(ACTIONS) - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_t).argmax().item()

            pyautogui.press(ACTIONS[action])

            # Active Window Validation Safety Check
            active_app = (
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to get name of first process whose frontmost is true',
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                .stdout.strip()
            )

            if "Google Chrome" not in active_app:
                print(
                    f"\nSAFETY HALT: Window focus shifted to '{active_app}'. Closing down."
                )
                print(
                    "Freezing state tensors and archiving network progress parameters..."
                )
                torch.save(policy_net.state_dict(), 'crossy_road_brain.pth')
                print("Progress saved successfully as 'crossy_road_brain.pth'")
                return

            time.sleep(0.05)
            next_state = get_ai_vision_state()
            
            # Updated: Fileless color matching execution check
            done = check_if_dead()

            if done:
                reward = -100.0
            elif action == 0:
                reward = 1.0
            else:
                reward = 0.1

            total_reward += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state

            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.FloatTensor(np.array(states)).to(device)
                actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_t = torch.FloatTensor(rewards).to(device)
                next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
                dones_t = torch.FloatTensor(dones).to(device)

                current_q = policy_net(states_t).gather(1, actions_t)
                max_next_q = target_net(next_states_t).max(1)[0].detach()
                expected_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))

                loss = nn.SmoothL1Loss()(current_q.squeeze(), expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(
            f"\nDeath Tracked! Episode: {episode} | Score Metric: {total_reward:.1f}"
        )
        print("Waiting for Game Over screen pop-up to settle down...")
        time.sleep(1.0)

        # Send the ENTER command to reset the run
        pyautogui.press('enter')
        print("Sent Restart Enter Key Command.")
        
        time.sleep(3.0)
        pyautogui.press('up')

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), 'crossy_road_brain.pth')
            print("Progress saved to 'crossy_road_brain.pth'")


if __name__ == "__main__":
    main()
