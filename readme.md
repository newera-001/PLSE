
# 1. How to Trains

## 1.1 Train Basic Skills
- Set `gait_explora_curriculum = False` in `legged_robot_config`, and set `parkour flat` to `1` in the `terrain_dict`.
- Training command:
```bash
python legged_gym/legged_gym/scripts/train.py --proj_name xx --exptid xx
```

## 1.2 Train Basic Skills for Parkour
- Set `parkour flat`, `parkour hurdle`, `parkour step`, and `parkour gap` all to `0.25` in the `terrain_dict`.
- Training command:
```bash
python legged_gym/legged_gym/scripts/train.py --proj_name xx --exptid xx --resume --resumeid xx --parkour_terrain
```

## 1.3 Start Learning to Explore New Skills
- Set `gait_explora_curriculum = True` in `legged_robot config`. Keep `parkour flat`, `parkour hurdle`, `parkour step`, and `parkour gap` all at `0.25` in the `terrain_dict`.
- Training command:
```bash
python legged_gym/legged_gym/scripts/train.py --proj_name xx --exptid xx --resume --resumeid xx --parkour_terrain
```

# 2. View Experimental Results
The pre-trained basic skills, basic skills for parkour, and explored skills are located in the projects named `basic_gait` and `explore_remeber`.

## 2.1 View Basic Skills
Commands:
```bash
python play.py --parkour_terrains --proj_name=basic_gait --resume --exptid=basic_skills
python play.py --parkour_terrains --proj_name=basic_gait --resume --exptid=basic_skill_parkour
```

## 2.2 View Newly Learned Skills
Command:
```bash
python play.py --parkour_terrains --proj_name=explore_remeber --resume --exptid=xx
```

# 3. Latent Variables of New Skills and Corresponding Files

## 3.1 Hopping
| Filename | Latent Variables | Skill | Other Feasible Latents |
| :--- | :--- | :--- | :--- |
| novelex1 | 3.87 -2.35 -3.46 -1.42 | Hopping | |

## 3.2 Possible Forefoot Standing
| Filename | Latent Variables | Skill | Other Feasible Latents |
| :--- | :--- | :--- |
| novelex1 | 2.03 2.39 -1.40 2.57 | Possible forefoot standing |

## 3.3 New Forms of Bound
| Filename | Latent Variables | Skill | Other Feasible Latents |
| :--- | :--- | :--- |
| novelex6 | -1.48 -3.15 -0.93 -1.133 | Aerial bound  |
| novelex6 | 1 1 0 0 | Transforms into a new form of bound |
| novelex5 | 1 0 1 1 | New skill: trot and bound fusion |
| novelex3 | 1 0 1 0 | Original motion disappears, becomes a mix of bound and trot |
| novelex2 | 1 0 1 0 | Becomes a new skill resembling bound |
| novelex2 | 1.11 -4.2 2.91 3 | Front crouched, hind legs raised while walking |
| novelex1 | 1 0 1 0 | New form of bound |
| novelex4 | 0 0 0 1 | Some kind of trot and bound mix |

## 3.4 Tripedal Walking
| Filename | Latent Variables | Skill | Other Feasible Latents |
| :--- | :--- | :--- |
| novelex6 | 1 0 0 1 | Original trot becomes a new skill: walking with right front leg lifted |
| novelex5 | 0 0 0 1 | New skill: front bound, rear trot |
| novelex3 | 0.21 2.16 1.23 0.83 | Left front leg lifted, tripedal walking |
| novelex2 | 2.72 3.10 -0.81 0.38 | Right front leg lifted |
| two_old_and_six_new | -1.37 -1 1 0.5 | Three‑legged walk | 1 0 0 1 (left rear) |
| two_old_and_six_new | -1.51 -4.73 -3.58 1.93 | (Right rear leg suspended) |
| two_old_and_six_new | 0 1.59 -1.71 -1.49 | (Right front leg disabled) |
| four2more | 0 0 1 0 | Walking with front legs lifted |
| four2more | 0.91 2.99 1.57 4.57 | Tripedal walking under pace: left front leg lifted high |

## 3.5 Bipedal Walking
| Filename | Latent Variables | Skill |
| :--- | :--- | :--- |
| novelex5 | 0.79 4.31 4.26 0.95 | Diagonal crawling |
| novelex3 | 1.68 2.72 -1.43 -3.18 | Front legs raised, bipedal | also 1.59 1.73 -5.38 4.66 |
| novelex2 | -0.88 2.5 -0.81 -2.42 | Walking sideways with left legs | -0.49 1.79 -3.91 -0.15 |
| two_old_and_six_new | -1.97 -2.38 1.15 1.84 | Rear legs lame, front legs moving |
| two_old_and_six_new | 1.97 2.38 -1.15 -1.84 | Front legs lame, rear legs moving |
| two_old_and_six_new | 0.31 4.39 7.02 0.84 | Diagonal pair disabled |
| four2more | 1.48 3.71 0.01 2.90 | Sideways walk, one side only |
| disparkour | -1.2 0.1 4.7 0.2 | Sideways walk |
