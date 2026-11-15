<p align="center">
  <img src="docs/figures/banner.png" alt="SeDriCa: autonomous golf cart, RoboRacer, vision transformers" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/team-SeDriCa%20·%20IIT%20Bombay-1f5fbf" alt="SeDriCa">
  <img src="https://img.shields.io/badge/RoboRacer-3rd%20place%20·%20Techfest%202025-f59e0b" alt="RoboRacer 3rd place">
  <img src="https://img.shields.io/badge/ROS%202-Humble-22314e" alt="ROS 2">
  <img src="https://img.shields.io/badge/stack-SLAM%20·%20particle%20filter%20·%20pure%20pursuit-ef4444" alt="stack">
  <img src="https://img.shields.io/badge/PyTorch-Vision%20Transformer-ee4c2c" alt="ViT">
</p>

> **TL;DR.** My work in **SeDriCa**, IIT Bombay's self-driving car team, at three scales:
> - a **full-size autonomous golf cart** driving on campus;
> - a **1/10-scale race car** that finished **3rd at the 26th International RoboRacer Autonomous Racing Competition** (Techfest 2025, IIT Bombay). Its stack is Cartographer mapping, particle-filter localisation and an adaptive pure-pursuit racing controller, with a mux that hands over to follow-the-gap for obstacle avoidance;
> - a **Vision Transformer** written from the paper for traffic-sign recognition.

---

## 1 · SeDriCa

**SeDriCa** (Self-Driving Car) is IIT Bombay's student team building autonomous vehicles end to end: sensors and drive-by-wire, mapping and localisation, planning and control, and perception. The same ideas run at two scales. On a **golf cart** they handle real campus roads; on a **1/10-scale F1TENTH / RoboRacer car** they are pushed to racing speed.

## 2 · The autonomous golf cart

<p align="center">
  <a href="docs/media/golf_cart_autonomous_drive.mp4"><img src="docs/figures/golf_cart_preview.gif" alt="The golf cart driving autonomously on the IIT Bombay campus, with the live LiDAR map on the laptop" width="75%"></a>
  <br><sub>The cart driving itself on campus. The laptop shows the live 3-D LiDAR map. <b><a href="docs/media/golf_cart_autonomous_drive.mp4">▶ full video (55 s)</a></b></sub>
</p>

```mermaid
flowchart LR
    L["3 × LiDAR<br/>top · left · right"] --> M["lidar_merger<br/>one point cloud<br/>(auto-calibrated extrinsics)"]
    M --> S["3-D LiDAR SLAM<br/>campus point-cloud map"]
    M --> N["NDT scan matching<br/>against the map"]
    G["GPS + IMU"] --> E["EKF<br/>(robot_localization)"]
    N --> E
    E --> P["Nav2<br/>planner + MPPI controller"]
    D["3-D detections"] --> X["emergency stop<br/>(object < 6 m in lane)"]
    P --> C["drive-by-wire<br/>(CAN)"]
    X --> C
    classDef s fill:#e8f0fc,stroke:#1f5fbf,color:#0b1a33
    classDef p fill:#e3f4e3,stroke:#2ca02c,color:#0e3b12
    classDef a fill:#fde8e8,stroke:#dc2626,color:#450a0a
    class L,G,D s
    class M,S,N,E,P p
    class X,C a
```

- **Seeing all around:** three LiDARs are fused into one cloud. [`lidar_merger`](golf_cart/mapping/lidar_merger) transforms each into the top sensor's frame using extrinsics from automatic LiDAR-to-LiDAR calibration ([`lidar_extrinsics/`](golf_cart/mapping/lidar_extrinsics)).
- **Knowing where it is:** a point-cloud map of the campus is built once with LiDAR SLAM ([`campus_map/`](golf_cart/mapping/campus_map): `map.pcd` + pose graph). Live scans are then matched against it with NDT, and the result is fused with GPS and IMU in an EKF.
- **Driving:** Nav2 plans the route and an **MPPI** controller tracks it ([`car_navigation/config`](golf_cart/navigation/car_navigation/config)). The stack was first brought up in the CARLA simulator through a Nav2 → CARLA adapter.
- **Safety and perception:** [`detection`](golf_cart/perception/detection) stops the cart if any detected object is closer than 6 m inside a 1 m-wide corridor ahead. A **YOLOv8 road-segmentation** model ([`road_segmentation/`](golf_cart/perception/road_segmentation)) was trained on about 8,000 road images and tested on highways and village roads.

## 3 · RoboRacer: autonomous racing

<p align="center">
  <img src="docs/photos/roboracer_certificate_3rd_place.jpeg" alt="Certificate: 3rd place, 26th International RoboRacer Autonomous Racing Competition, Techfest 2025, IIT Bombay" width="60%">
  <br><sub><b>3rd place</b>, 26th International RoboRacer Autonomous Racing Competition, Techfest 2025, IIT Bombay.</sub>
</p>

<table>
<tr>
<td width="33%"><img src="docs/photos/roboracer_team.jpeg" alt="The team with the race car at the competition"><br><sub>The team with our car at the venue</sub></td>
<td width="33%"><img src="docs/photos/roboracer_car_on_track.jpeg" alt="The car racing on the track"><br><sub>Racing on the track</sub></td>
<td width="33%"><img src="docs/photos/roboracer_track.jpeg" alt="The competition track"><br><sub>The competition track</sub></td>
</tr>
</table>

RoboRacer (formerly F1TENTH) races 1/10-scale cars that must drive fully autonomously with only on-board sensing: a 2-D LiDAR, wheel odometry and an IMU. Our stack:

```mermaid
flowchart LR
    Li["2-D LiDAR"] --> C["Cartographer<br/>build the track map<br/>(before the race)"]
    C --> Map[("occupancy map")]
    Li --> PF["Particle filter<br/>where am I on the map?"]
    O["odometry + IMU"] --> PF
    Map --> PF
    PF --> A["ALPP<br/>adaptive pure pursuit<br/>on the raceline"]
    Li --> F["FTG<br/>follow the gap"]
    Li --> Det["obstacle check<br/>scan points not in the map"]
    A --> MX{{"mux"}}
    F --> MX
    Det -. selects .-> MX
    MX --> V["VESC<br/>speed + steering"]
    classDef s fill:#e8f0fc,stroke:#1f5fbf,color:#0b1a33
    classDef p fill:#e3f4e3,stroke:#2ca02c,color:#0e3b12
    classDef c fill:#fff4dc,stroke:#e0a100,color:#3d2c00
    class Li,O s
    class C,Map,PF p
    class A,F,Det,MX,V c
```

The three animations below run on **our actual competition map** ([`roboracer/maps/competition_track.pgm`](roboracer/maps), 1.2 cm per pixel). The car and its LiDAR are simulated, and each algorithm follows our code and parameters.

### Mapping with Cartographer

Before racing, the car is driven around the track once while Cartographer builds a 2-D occupancy grid. Every LiDAR beam marks the cells it passes through as free and its end point as a wall. Scan by scan, the track appears:

<p align="center"><img src="docs/figures/mapping.gif" alt="Animation: the occupancy grid of the race track growing scan by scan as the car drives a lap" width="70%"></p>

Our Cartographer setups for mapping with LiDAR alone, LiDAR + IMU, and localisation in a saved map are in [`roboracer/mapping/cartographer`](roboracer/mapping/cartographer).

### Localisation with a particle filter

During the race the car must keep track of where it is on that map, many times per second. A **particle filter** keeps thousands of guesses (particles) of the car's pose:
1. **predict:** move every particle by the odometry, with noise;
2. **weight:** project the real LiDAR scan from each particle's pose. A particle scores high if the beam end-points land on walls of the map (a *likelihood field*);
3. **resample:** keep the likely particles and drop the unlikely ones.

Started with no idea where the car is, the red particles cover the whole map, split into a few competing clusters wherever the track looks alike, then lock onto the car:

<p align="center"><img src="docs/figures/particle_filter.gif" alt="Animation: red particles spread over the whole map collapsing onto the car's true position" width="70%"></p>

### Racing with adaptive pure pursuit (ALPP), and follow-the-gap when something is in the way

**ALPP** ([`roboracer/controller/adaptive_pure_pursuit.py`](roboracer/controller/adaptive_pure_pursuit.py)) chases a point on the raceline a *look-ahead* distance ahead of the car:
- **steering:** δ = atan(2L·sin α / L_d), where α is the angle to the goal point and L = 0.33 m is the wheelbase;
- **look-ahead:** grows with speed along a sigmoid, so the car aims further ahead on straights and tighter in corners;
- **speed:** a blend β·v_curv + (1−β)·v_dev of a curvature-limited speed and a speed that drops when the car has drifted off the line. The weight β adapts online.

The **mux** passes ALPP's commands to the motors until the LiDAR sees **points that are not in the map**: an obstacle on the racing line. It then switches to **follow-the-gap (FTG)**:
1. put a "bubble" around the closest point;
2. find the widest stretch of free space in the scan;
3. steer into it.

Once the way is clear, the mux hands back to ALPP and the car rejoins the line:

<p align="center"><img src="docs/figures/alpp_ftg_mux.gif" alt="Animation: the car follows the raceline with ALPP, the mux switches to FTG to pass an obstacle, then back to ALPP" width="100%"></p>

## 4 · Vision Transformer for traffic signs

A self-driving car has to read traffic signs. I implemented the **Vision Transformer** from *An Image is Worth 16×16 Words* (Dosovitskiy et al., ICLR 2021) in PyTorch, written module by module from the paper's equations ([`vision_transformer/`](vision_transformer)), and used it for traffic-sign classification (43 classes, GTSRB format).

**From pixels to tokens.** A Transformer reads a sequence of "words", so the image is turned into one:

<p align="center"><img src="docs/figures/vit_patch_embedding.gif" alt="Animation: a speed-limit sign cut into 16x16 patches, lined up into a sequence, flattened, projected to embeddings, with class token and position embeddings added" width="100%"></p>

**Attention between patches.** In every layer, each patch compares itself with every other patch (query · key), turns the scores into weights with a softmax, and takes a weighted mix of their values. Different heads learn different habits. Some attend **locally**, to neighbouring patches, much like a convolution. Others attend **globally**, to patches anywhere in the image with related content, such as all the pieces of a sign's red ring. Here, the highlighted patch is the query and brightness is its attention to each patch:

<p align="center"><img src="docs/figures/vit_attention.gif" alt="Animation: attention from a query patch, a local head attending to neighbours and a global head attending to similar patches across the sign" width="100%"></p>

<sub>The attention weights in this animation are illustrative: they are built from patch colour, texture and distance to show the two behaviours. A trained ViT learns its own query and key projections. <code>predict.py</code> draws the real attention of a trained model with attention rollout.</sub>

| Paper | Code ([`vit/model.py`](vision_transformer/vit/model.py)) |
|---|---|
| (1) z₀ = [x_class ; x_p¹E ; … ; x_pᴺE] + E_pos | `PatchEmbedding` (Conv2d, kernel = stride = 16), `cls_token`, `pos_embed` |
| (2) z′ = MSA(LN(z)) + z | `EncoderBlock`, attention half; `MultiHeadSelfAttention` written by hand, returns its weights |
| (3) z = MLP(LN(z′)) + z′ | `EncoderBlock`, MLP half (GELU) |
| (4) y = LN(z_L⁰) | final `LayerNorm` on the class token → linear head |
| ViT-Base: L = 12, D = 768, 12 heads, MLP 3072 | `vit_base()`: **85.8 M parameters** (paper: 86 M) |

```bash
cd vision_transformer
python -m vit.model                 # shape check: ViT-Base/16 and ViT-Tiny/8
python train.py                     # ViT-Tiny/8 on 64×64 GTSRB traffic signs (AdamW, warm-up + cosine, label smoothing)
python predict.py checkpoints/vit_best.pt sign.png   # top-3 classes + attention-rollout map
```

---

## Repository

```
golf_cart/            the team's packages for the autonomous golf cart (ROS 2)
roboracer/            race-car controller, particle filter, Cartographer configs and the competition map
vision_transformer/   ViT written from the paper: model, data pipeline, training, prediction
docs/                 photos, video, figures and the scripts that make the animations
```

<details open>
<summary><b>What every folder holds</b></summary>

| Folder | What it is |
|---|---|
| `golf_cart/mapping/lidar_merger` | C++ node fusing the three LiDARs into one point cloud |
| `golf_cart/mapping/lidar_extrinsics`, `campus_map` | LiDAR-to-LiDAR calibration results; campus point-cloud map and pose graph |
| `golf_cart/mapping/bag_to_pcd.py` | export point clouds from a rosbag |
| `golf_cart/navigation/car_navigation` | Nav2 configuration (MPPI controller, costmaps), behaviour tree, launch files, LiDAR-to-scan and CARLA adapters |
| `golf_cart/navigation/vehicle_iitb` | 2-D map of the test area for Nav2 |
| `golf_cart/perception/detection` | emergency-stop node on 3-D detections |
| `golf_cart/perception/road_segmentation` | YOLOv8 road-segmentation model (`best.pt`) and its training notebook |
| `roboracer/controller` | `adaptive_pure_pursuit.py` (ALPP), and a simulator raceline |
| `roboracer/localization/particle_filter` | particle filter (based on the open-source JMU ROS 2 particle filter, MIT licence) and its configs |
| `roboracer/mapping/cartographer` | Cartographer configs and launch files: scan-only mapping, scan + IMU, localisation |
| `roboracer/maps` | the competition track map |
| `vision_transformer/` | `vit/model.py`, `vit/data.py`, `train.py`, `predict.py` |
| `docs/` | `make_figures.py` + `track_sim.py` (RoboRacer animations), `vit_figures.py` (ViT animations), `media_figures.py` (banner, video preview) |

</details>

> [!NOTE]
> Third-party building blocks are used as published and are not copied here: Cartographer, Nav2, robot_localization, the LiDAR SLAM and NDT packages, the Ouster/Velodyne drivers, and the vehicle's drive-by-wire interface.

## Acknowledgements

SeDriCa, IIT Bombay. RoboRacer Foundation and Techfest IIT Bombay for the competition. ViT: A. Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. Attention rollout: S. Abnar and W. Zuidema, ACL 2020.
