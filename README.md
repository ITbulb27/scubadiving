**ITBULB**

**Owner:** ITbulb owner

**Overview**
- **Purpose:** Simple Python project to run a hand-landmark/video demo using a hand landmarker task file.
- **Quick:** run the main script to start the camera and see hand landmark output.

**Requirements**
- **Python:** 3.8+ recommended.
- **Packages:** OpenCV and MediaPipe (or the packages your [hand_landmarker.task](hand_landmarker.task) expects).

**Setup**
- **Create venv:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- **Install dependencies:**

```powershell
pip install -r requirements.txt
# or, if you don't have requirements.txt:
pip install opencv-python mediapipe
```

**Run**
- **Start the app:**

```powershell
python mains.py
```

**Files**
- **Main script:** [mains.py](mains.py) — entry point that opens the camera and runs the hand landmark demo.
- **Task file:** [hand_landmarker.task](hand_landmarker.task) — task/config used by the landmarker model.

**Notes**
- If your project uses a different model or package, update requirements.txt accordingly.
- On Windows, use the PowerShell venv activation command shown above.

**Credits**
- Created for quick local hand-landmarker testing. Ask me to add examples, screenshots, or a requirements file.
