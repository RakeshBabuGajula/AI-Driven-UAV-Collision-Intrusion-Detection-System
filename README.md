<div align="center">

# 🚁 AI-Driven UAV Collision & Intrusion Detection System
### *Autonomous Airspace Deconfliction, Deep Learning Perception & Geospatial Geofencing*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)



[Key Capabilities](#-key-capabilities) •
[System Architecture](#-system-architecture) •
[Kinematic Math & Algorithms](#-kinematic-formulation--conflict-algorithms) •
[Mission Control UI](#-mission-control--visual-analytics) •
[Quickstart Guide](#-quickstart--installation) •
[Technical Highlights](#-resume--engineering-highlights) •
[Author & Contact](#-author--contact)

---

</div>

## 📌 Executive Summary

As low-altitude airspace becomes increasingly crowded with commercial delivery drones, autonomous inspection UAVs, and hobbyist aircraft, the risk of **mid-air collisions** and **unauthorized incursions into restricted airspace** grows exponentially. Traditional Air Traffic Management (ATM) relying on transponders and ground radar is ill-suited for dense, low-altitude Unmanned Traffic Management (UTM).

The **AI-Driven UAV Collision & Intrusion Detection System** is an end-to-end, edge-ready UTM solution engineered to ensure safe airspace coexistence. By unifying **real-time deep learning computer vision (YOLOv8)**, **3D predictive relative kinematics (Closest Point of Approach - CPA)**, and **computational geometry geofencing (Point-in-Polygon)**, this system continuously detects aerial threats, forecasts collision risks before they happen, and alerts operators through a reactive **Streamlit Mission Control Console**.

---

## ⚡ Key Capabilities

| Capability | Technical Implementation | Impact / Outcome |
| :--- | :--- | :--- |
| **Vision-Based Aerial Perception** | YOLOv8 Deep Convolutional Neural Network | Real-time aerial object & intruder detection on live video feeds with sub-50ms inference. |
| **Predictive Conflict Detection (CD&R)** | 3D Kinematic Closest Point of Approach (CPA) | Analytically calculates time-to-closest-approach ($t_{ca}$) and minimum separation distance ($d_{min}$) in future trajectory windows. |
| **Dynamic Airspace Geofencing** | Computational Geometry via Shapely (`Point-in-Polygon`) | Enforces custom 2D/3D no-fly zones (NFZs) and triggers microsecond security containment alerts. |
| **Tactical Mission Control Console** | Streamlit Reactive Operations Dashboard | Provides operators with live alerting cards, temporal separation charts, and intrusion frequency analytics. |
| **Resilient Event Streaming & Audit Trails** | Non-blocking structured JSON Lines (`JSONL`) logging | High-throughput logging of collision metrics and security events for post-mission black-box analysis. |

---

## 🏗️ System Architecture

The end-to-end data processing pipeline flows from sensory input and trajectory generation to real-time analysis, alerting, and tactical dashboard visualization:

```mermaid
flowchart TD
    subgraph SENSORS["1. Ingestion & Perception Layer"]
        A1[Live Video / Camera Feed] --> B1[YOLOv8 Detection Engine]
        A2[UAV Telemetry Stream x, y, z, t] --> B2[Kinematic State Estimator]
        B1 --> C1[Aerial Object Bounding Boxes & Confidence]
    end

    subgraph ENGINE["2. Spatial-Temporal Analytics Engine"]
        B2 --> D1[3D Velocity Vector Estimation]
        D1 --> E1[Closest Point of Approach Engine]
        E1 -->|Compute t_ca & d_min| F1{Separation Threat?}
        F1 -->|Distance < Threshold| G1[Trigger Collision ALERT]
        F1 -->|Distance >= Threshold| G2[Status: Clear / OK]

        B2 --> D2[Geospatial Geofencing Engine]
        D2 -->|Point-in-Polygon Check| F2{Boundary Breach?}
        F2 -->|Inside Restricted Zone| G3[Trigger Intrusion ALERT]
        F2 -->|Outside Zone| G4[Status: Normal Airspace]
    end

    subgraph LOGGING["3. Event Streaming & Telemetry Logs"]
        G1 & G2 --> H1[(collision.jsonl)]
        G3 & G4 --> H2[(intrusion.jsonl & alerts.json)]
    end

    subgraph UI["4. Mission Control Operations Console"]
        H1 --> I1[Streamlit Reactive Engine]
        H2 --> I1
        I1 --> J1[Live Hazard Alert Banners]
        I1 --> J2[Temporal Distance Trend Graph]
        I1 --> J3[Airspace Intrusion Bar Chart]
        I1 --> J4[2D Spatial Flight Path Plots]
    end
```

---

## 📐 Kinematic Formulation & Conflict Algorithms

### 1. 3D Closest Point of Approach (CPA)
To prevent mid-air collisions without relying on external radar, the system calculates the relative kinematics between two UAVs moving with continuous position vectors $\vec{p}_1(t), \vec{p}_2(t)$ and estimated velocity vectors $\vec{v}_1, \vec{v}_2$:

$$\vec{\Delta p} = \vec{p}_1(0) - \vec{p}_2(0)$$

$$\vec{\Delta v} = \vec{v}_1 - \vec{v}_2$$

The relative separation vector as a function of time $t$ is:

$$\vec{d}(t) = \vec{\Delta p} + \vec{\Delta v} \cdot t$$

Squaring the distance gives $D^2(t) = \|\vec{d}(t)\|^2 = \|\vec{\Delta p}\|^2 + 2(\vec{\Delta p} \cdot \vec{\Delta v})t + \|\vec{\Delta v}\|^2 t^2$.

Minimizing $D^2(t)$ with respect to $t$ yields the **Time of Closest Approach ($t_{ca}$)**:

$$t_{ca} = \max\left(0, -\frac{\vec{\Delta p} \cdot \vec{\Delta v}}{\|\vec{\Delta v}\|^2}\right)$$

Evaluating the positions at $t_{ca}$ gives the **Minimum Separation Distance ($d_{min}$)**:

$$d_{min} = \left\| (\vec{p}_1(0) + \vec{v}_1 t_{ca}) - (\vec{p}_2(0) + \vec{v}_2 t_{ca}) \right\|$$

- If $d_{min} < D_{\text{threshold}}$, an immediate **`ALERT`** is broadcast with exact intercept time.
- If $d_{min} \ge D_{\text{threshold}}$, the conflict risk is logged as **`OK`**.

### 2. Computational Geometry Geofencing
Restricted airspace polygons $\Omega \subset \mathbb{R}^2$ are defined via vertex sequences $V = \{v_1, v_2, \dots, v_k\}$. For every telemetry ping $(x_u, y_u)$, the system applies a Jordan curve ray-casting algorithm via `Shapely`:

$$\text{Intrusion}(u) = \begin{cases} 
\text{True} & \text{if } (x_u, y_u) \in \text{Interior}(\Omega) \\
\text{False} & \text{otherwise}
\end{cases}$$

Breaches immediately dispatch structured telemetry to `alerts.json` and `intrusion.jsonl`.

---

## 🖥️ Mission Control & Visual Analytics

### 1. Real-Time Operations Console (Streamlit)
Live dashboard providing automated state polling, severity-coded status cards, and temporal separation tracking.

| Live Status & Trend Analysis | Intrusion Frequency Monitoring |
| :---: | :---: |
| <img width="100%" alt="Status and Collision Alerts" src="https://github.com/user-attachments/assets/f39594de-97fd-4ea4-9af3-d62a5abc085e" /> | <img width="100%" alt="Charts and Intrusion Status" src="https://github.com/user-attachments/assets/7cb1e25f-9fef-42cb-bbf0-1089fd9a4a35" /> |
| *Real-time collision alert cards & distance-over-time trendline* | *Intrusion frequency over rolling temporal frames* |

### 2. 2D Spatial Airspace Deconfliction & Intrusion Maps
High-resolution kinematic trajectories showing flight paths, restricted no-fly zones, and flagged violation coordinates.

| Kinematic Approach Trajectories | Geofence Boundary Breach Flagging |
| :---: | :---: |
| <img width="100%" alt="UAV Flight Paths" src="https://github.com/user-attachments/assets/8ed09f92-cada-4af6-ab3c-b88f741769e7" /> | <img width="100%" alt="UAV Paths with Alerts" src="https://github.com/user-attachments/assets/57eb5bbe-078e-4574-b65b-18026e9cd606" /> |
| *2D flight path intersection vectors for UAV 1 & UAV 2* | *Shapely geofence polygon with real-time breach triggers (`x`)* |

---

## 🗂️ Project Structure

```bash
AI-Driven-UAV-Collision-Intrusion-Detection-System/
├── README.md                  # Comprehensive technical documentation & portfolio overview
├── LICENSE                    # MIT Open-Source License
├── documentation.md           # Extended system API & technical specifications
├── yolov8n.pt                 # Pretrained YOLOv8 Nano weights for low-latency detection
│
└── uav-safety/
    ├── streamlit_app.py       # Tactical Mission Control Dashboard (Streamlit)
    ├── requirements.txt       # Core project dependencies & libraries
    │
    ├── scripts/
    │   ├── check_collision.py # 3D CPA kinematic solver & minimum distance predictor
    │   ├── visualize.py       # Shapely geofence intrusion checker & 2D spatial plotter
    │   ├── generate_flights.py# Synthetic 3D flight trajectory generator
    │   ├── dataset.py         # Advanced multi-UAV GPS/altitude telemetry generator
    │   ├── run_yolo_demo.py   # Full YOLOv8 video & image inference pipeline
    │   └── yolo_demo.py       # Lightweight drone/aerial object detector
    │
    ├── data/
    │   ├── flight1.csv        # Telemetry stream for UAV 1 (t, x, y, z)
    │   ├── flight2.csv        # Telemetry stream for UAV 2 (t, x, y, z)
    │   ├── alerts.json        # Structured JSON record of active geofence violations
    │   └── sample.mp4         # Sample video stream for computer vision validation
    │
    ├── logs/
    │   ├── collision.jsonl    # High-frequency conflict logs (timestamps, distances, risks)
    │   └── intrusion.jsonl    # High-frequency geofence security logs (status, IDs, coords)
    │
    └── runs/
        ├── flight_paths.png   # 2D kinematic trajectory output
        ├── paths.png          # Geofence boundary & breach coordinate visualization
        └── detect/            # YOLOv8 annotated video & detection bounding boxes
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python 3.10+**
- **Git**
- Optional: CUDA-compatible GPU (for accelerated YOLO inference)

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/GajulaRakeshBabu/AI-Driven-UAV-Collision-Intrusion-Detection-System.git
cd AI-Driven-UAV-Collision-Intrusion-Detection-System/uav-safety

# Create and activate virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Verification Pipeline

#### A. Run YOLOv8 Perception on Video
```bash
python scripts/run_yolo_demo.py
```
*Processes `data/sample.mp4`, performs neural detection, and outputs annotated video to `runs/detect/`.*

#### B. Synthesize Trajectories & Check 3D Collisions
```bash
python scripts/generate_flights.py
python scripts/check_collision.py
```
*Outputs Closest Approach Time ($t_{ca}$) and Minimum Distance ($d_{min}$) in meters.*

#### C. Perform Geofence Intrusion Detection
```bash
python scripts/visualize.py
```
*Evaluates points against the polygon geofence, writes alerts to `data/alerts.json`, and generates `runs/paths.png`.*

#### D. Launch Streamlit Tactical Dashboard
```bash
streamlit run streamlit_app.py
```
*Open [http://localhost:8501](http://localhost:8501) in your browser to view the live dashboard.*

---

## 🛠️ Tech Stack & Engineering Tools

| Category | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Deep Learning & CV** | `Ultralytics YOLOv8`, `OpenCV` | Real-time object detection, bounding box extraction, video stream parsing. |
| **Computational Geometry** | `Shapely` | Polygon boundary definition, point-in-polygon spatial containment. |
| **Kinematic Modeling** | `NumPy`, `Pandas` | 3D vector algebra, trajectory generation, CPA analytical solver. |
| **Tactical Dashboard** | `Streamlit` | Auto-refreshing reactive UI, live hazard cards, dynamic time-series charts. |
| **Spatial Visualization** | `Matplotlib` | High-fidelity coordinate maps, approach vectors, alert overlays. |
| **Data Streaming** | `JSON Lines (JSONL)` | Low-latency append-only telemetry logging for mission review. |

---

## Highlights

This project demonstrates core competencies across multiple engineering domains:

- **Computer Vision & Deep Learning**: Integrated state-of-the-art YOLOv8 architecture into an automated video inference pipeline, extracting bounding boxes, confidence values, and object classifications.
- **Aerospace & Applied Kinematics**: Implemented real-time relative motion kinematics (Closest Point of Approach) to forecast potential UAV mid-air collisions quantitatively rather than reactively.
- **Computational Geometry & GIS**: Designed dynamic geofencing algorithms using polygonal spatial analysis (`Shapely`), enforcing security zones and no-fly zones (NFZs) with coordinate precision.
- **Full-Stack Tactical Dashboard**: Engineered a reactive, auto-updating operations dashboard in Streamlit that digests asynchronous JSONL log streams to display real-time safety statuses and analytical charts.
- **Software Engineering Best Practices**: Built with clean separation of concerns, modularized analytical scripts, structured JSON/JSONL logging, and comprehensive documentation.

---

## 🔮 Roadmap & Future Enhancements

- [ ] **Extended Kalman Filtering (EKF)**: Integrate sensor fusion (IMU + GPS + Vision) for robust state estimation in noisy environments.
- [ ] **Multi-Agent Swarm Deconfliction**: Scale the CPA algorithm from pairwise UAVs to $N$-UAV swarms using spatial partitioning (KD-Trees/Octrees).
- [ ] **ROS2 / PX4 Autopilot Integration**: Bridge collision warnings directly to MAVLink / PX4 companion computers for automated evasive maneuvers.
- [ ] **Edge Deployment**: Benchmark and package as a lightweight TensorRT engine for NVIDIA Jetson Orin Nano companion computers.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

---

## 👨‍💻 Author & Contact

**Gajula Rakesh Babu**  
*AI & Autonomous Systems Enthusiast | Software Engineer*

- **GitHub**: [@GajulaRakeshBabu](https://github.com/GajulaRakeshBabu)
- **Email**: [grakesh2605@gmail.com](mailto:grakesh2605@gmail.com)
- **LinkedIn**: [linkedin.com/in/rakeshbabugajula](https://www.linkedin.com/in/rakeshbabugajula/)

---

<div align="center">

*⭐ If this project provided value to you or your research, consider giving it a star on GitHub!*

</div>
