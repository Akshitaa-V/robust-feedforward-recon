### AbsRel by condition (severity 2; lower is better; mean over seeds)

| Method | clean | fog | rain | snow (unseen) | low_light | shadows | occlusion | Mean corrupted |
|---|---|---|---|---|---|---|---|---|
| Single frame, clean training | 0.026 | 0.341 | 0.154 | 0.356 | 0.502 | 0.173 | 0.099 | 0.271 |
| 3 frames, clean training | 0.026 | 0.248 | 0.131 | 0.280 | 0.442 | 0.138 | 0.103 | 0.224 |
| 3 frames + corruption augmentation | 0.034 | 0.029 | 0.034 | 0.068 | 0.057 | 0.045 | 0.069 | 0.050 |
| 3 frames + aug + photometric norm + consistency | 0.028 | 0.027 | 0.032 | 0.052 | 0.052 | 0.036 | 0.063 | 0.044 |

### All metrics, averaged over the six corruptions (severity 2)

| Method | AbsRel | RMSE (m) | delta<1.25 | AbsRel moving objects | Temporal inconsistency | Chamfer (m) | BEV occupancy IoU |
|---|---|---|---|---|---|---|---|
| Single frame, clean training | 0.271 | 8.738 | 0.710 | 0.769 | 0.166 | 0.794 | 0.152 |
| 3 frames, clean training | 0.224 | 7.720 | 0.761 | 0.726 | 0.136 | 0.710 | 0.164 |
| 3 frames + corruption augmentation | 0.050 | 2.476 | 0.962 | 0.179 | 0.033 | 0.322 | 0.288 |
| 3 frames + aug + photometric norm + consistency | 0.044 | 2.341 | 0.969 | 0.148 | 0.032 | 0.303 | 0.303 |

### Clean images

| Method | AbsRel | RMSE (m) | delta<1.25 | AbsRel moving objects | Temporal inconsistency | Chamfer (m) | BEV occupancy IoU |
|---|---|---|---|---|---|---|---|
| Single frame, clean training | 0.026 | 1.589 | 0.985 | 0.064 | 0.024 | 0.252 | 0.347 |
| 3 frames, clean training | 0.026 | 1.559 | 0.987 | 0.064 | 0.022 | 0.247 | 0.360 |
| 3 frames + corruption augmentation | 0.034 | 1.867 | 0.976 | 0.090 | 0.024 | 0.284 | 0.307 |
| 3 frames + aug + photometric norm + consistency | 0.028 | 1.622 | 0.983 | 0.072 | 0.022 | 0.254 | 0.342 |

### Severity sweep (AbsRel)

| Condition | Method | 1 | 2 | 3 |
|---|---|---|---|---|
| fog | Single frame, clean training | 0.088 | 0.341 | 0.738 |
| fog | 3 frames, clean training | 0.067 | 0.248 | 0.547 |
| fog | 3 frames + corruption augmentation | 0.029 | 0.029 | 0.033 |
| fog | 3 frames + aug + photometric norm + consistency | 0.026 | 0.027 | 0.033 |
| low_light | Single frame, clean training | 0.401 | 0.502 | 0.591 |
| low_light | 3 frames, clean training | 0.296 | 0.442 | 0.585 |
| low_light | 3 frames + corruption augmentation | 0.043 | 0.057 | 0.094 |
| low_light | 3 frames + aug + photometric norm + consistency | 0.035 | 0.052 | 0.090 |
| snow | Single frame, clean training | 0.170 | 0.356 | 0.550 |
| snow | 3 frames, clean training | 0.157 | 0.280 | 0.453 |
| snow | 3 frames + corruption augmentation | 0.055 | 0.068 | 0.085 |
| snow | 3 frames + aug + photometric norm + consistency | 0.043 | 0.052 | 0.060 |
