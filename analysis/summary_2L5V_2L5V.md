# Experiment Results Summary

## 2L5V 2L5V Results


### Main Experimental Results

| Method | Episode Length (s) | Success Rate (%) | Collision Rate (hr⁻¹) | Speed (m/s) | Cost Rate (hr⁻¹) | Avoided Cost Rate (hr⁻¹) | Left Lane Preference (%) | Right Lane Preference (%) |
|---|---|---|---|---|---|---|---|---|
| **Cautious Profile** |  |  |  |  |  |  |  |  |
| Unsupervised | 26.98 ± 0.19 | 82.40 | 23.49 ± 2.27 | 28.04 ± 0.10 | 5689.21 ± 28.54 | 0.00 ± 0.00 | 45.80 ± 0.36 | 54.20 ± 0.36 |
| Filter_Only | 26.78 ± 0.42 | 85.20 | 19.90 ± 2.14 | 27.85 ± 0.10 | 5464.89 ± 42.61 | 213.34 ± 25.39 | 46.17 ± 0.44 | 53.83 ± 0.44 |
| Naive | 29.45 ± 0.22 | 96.80 | 3.91 ± 0.96 | 25.19 ± 0.10 | 2392.58 ± 47.79 | 3040.46 ± 30.19 | 27.80 ± 0.62 | 72.20 ± 0.62 |
| Adaptive (10⁻²) | 29.68 ± 0.16 | 98.00 | 2.43 ± 0.76 | 24.26 ± 0.08 | 1738.05 ± 58.26 | 3888.02 ± 26.54 | 30.71 ± 1.17 | 69.29 ± 1.17 |
| Adaptive (10⁻¹) | 29.69 ± 0.08 | 97.60 | 2.91 ± 0.83 | 22.36 ± 0.07 | 206.47 ± 17.54 | 4722.73 ± 64.09 | 5.79 ± 0.51 | 94.21 ± 0.51 |
| Adaptive (10⁰) | 29.73 ± 0.05 | 97.80 | 2.66 ± 0.79 | 22.28 ± 0.07 | 125.66 ± 12.10 | 4717.79 ± 53.73 | 4.55 ± 0.40 | 95.45 ± 0.40 |
| Fixed (10⁻²) | 29.73 ± 0.05 | 97.80 | 2.66 ± 0.79 | 22.28 ± 0.07 | 125.66 ± 12.10 | 4717.79 ± 53.73 | 4.55 ± 0.40 | 95.45 ± 0.40 |
| Fixed (10⁻¹) | 29.73 ± 0.05 | 97.80 | 2.66 ± 0.79 | 22.28 ± 0.07 | 125.66 ± 12.10 | 4717.79 ± 53.73 | 4.55 ± 0.40 | 95.45 ± 0.40 |
| Fixed (10⁰) | 29.57 ± 0.24 | 97.20 | 3.41 ± 0.90 | 24.35 ± 0.09 | 1723.11 ± 64.06 | 3737.74 ± 56.65 | 27.42 ± 0.99 | 72.58 ± 0.99 |
| Projection | 29.73 ± 0.05 | 97.80 | 2.66 ± 0.79 | 22.28 ± 0.07 | 125.66 ± 12.10 | 4717.79 ± 53.73 | 4.55 ± 0.40 | 95.45 ± 0.40 |
|  |  |  |  |  |  |  |  |  |
| **Efficient Profile** |  |  |  |  |  |  |  |  |
| Unsupervised | 26.98 ± 0.19 | 82.40 | 23.49 ± 2.27 | 28.04 ± 0.10 | 2696.39 ± 22.31 | 0.00 ± 0.00 | 45.80 ± 0.36 | 54.20 ± 0.36 |
| Filter_Only | 26.78 ± 0.42 | 85.20 | 19.90 ± 2.14 | 27.85 ± 0.10 | 2485.45 ± 21.12 | 316.18 ± 21.55 | 46.17 ± 0.44 | 53.83 ± 0.44 |
| Naive | 27.88 ± 0.26 | 90.40 | 12.40 ± 1.70 | 27.43 ± 0.07 | 1187.88 ± 21.45 | 1492.21 ± 28.64 | 73.64 ± 0.70 | 26.36 ± 0.70 |
| Adaptive (10⁻²) | 28.46 ± 0.23 | 93.40 | 8.35 ± 1.40 | 27.09 ± 0.06 | 1049.07 ± 30.50 | 1601.63 ± 25.30 | 74.71 ± 0.91 | 25.29 ± 0.91 |
| Adaptive (10⁻¹) | 28.51 ± 0.13 | 93.40 | 8.34 ± 1.40 | 24.83 ± 0.05 | 464.99 ± 41.65 | 2313.63 ± 32.64 | 86.61 ± 1.08 | 13.39 ± 1.08 |
| Adaptive (10⁰) | 28.36 ± 0.19 | 92.80 | 9.14 ± 1.47 | 23.46 ± 0.07 | 252.78 ± 21.35 | 2726.06 ± 54.64 | 92.24 ± 0.56 | 7.76 ± 0.56 |
| Fixed (10⁻²) | 28.32 ± 0.21 | 92.60 | 9.41 ± 1.49 | 23.46 ± 0.07 | 251.02 ± 23.10 | 2723.32 ± 53.29 | 92.29 ± 0.60 | 7.71 ± 0.60 |
| Fixed (10⁻¹) | 28.36 ± 0.19 | 92.80 | 9.14 ± 1.47 | 23.46 ± 0.08 | 254.55 ± 21.93 | 2724.79 ± 54.38 | 92.19 ± 0.57 | 7.81 ± 0.57 |
| Fixed (10⁰) | 28.24 ± 0.29 | 92.20 | 9.94 ± 1.53 | 27.18 ± 0.03 | 1060.30 ± 21.75 | 1606.31 ± 28.39 | 75.25 ± 0.70 | 24.75 ± 0.70 |
| Projection | 28.32 ± 0.21 | 92.60 | 9.41 ± 1.49 | 23.46 ± 0.07 | 251.02 ± 23.10 | 2723.32 ± 53.29 | 92.29 ± 0.60 | 7.71 ± 0.60 |

### Ablation Studies

| Method | Episode Length (s) | Success Rate (%) | Collision Rate (hr⁻¹) | Speed (m/s) | Cost Rate (hr⁻¹) | Avoided Cost Rate (hr⁻¹) | Left Lane Preference (%) | Right Lane Preference (%) |
|---|---|---|---|---|---|---|---|---|
| **Cautious Profile** |  |  |  |  |  |  |  |  |
| Naive | 29.35 ± 0.14 | 95.00 | 6.13 ± 1.20 | 25.29 ± 0.08 | 2405.67 ± 24.51 | 3023.09 ± 15.31 | 26.65 ± 0.40 | 73.35 ± 0.40 |
| Adaptive (10⁻²) | 29.34 ± 0.22 | 95.60 | 5.40 ± 1.13 | 24.36 ± 0.08 | 1753.90 ± 43.41 | 3851.88 ± 36.73 | 29.10 ± 0.96 | 70.90 ± 0.96 |
| Adaptive (10⁻¹) | 29.50 ± 0.16 | 96.60 | 4.15 ± 0.99 | 22.36 ± 0.07 | 202.35 ± 15.10 | 4724.11 ± 62.34 | 5.57 ± 0.51 | 94.43 ± 0.51 |
| Adaptive (10⁰) | 29.52 ± 0.11 | 96.60 | 4.15 ± 0.99 | 22.28 ± 0.07 | 113.33 ± 11.50 | 4720.90 ± 51.78 | 4.24 ± 0.41 | 95.76 ± 0.41 |
| Fixed (10⁻²) | 29.52 ± 0.11 | 96.60 | 4.15 ± 0.99 | 22.28 ± 0.07 | 113.33 ± 11.50 | 4720.90 ± 51.78 | 4.24 ± 0.41 | 95.76 ± 0.41 |
| Fixed (10⁻¹) | 29.52 ± 0.11 | 96.60 | 4.15 ± 0.99 | 22.28 ± 0.07 | 116.01 ± 12.96 | 4716.16 ± 53.56 | 4.32 ± 0.45 | 95.68 ± 0.45 |
| Fixed (10⁰) | 29.36 ± 0.14 | 95.60 | 5.40 ± 1.12 | 24.79 ± 0.09 | 1960.53 ± 57.90 | 3460.56 ± 46.34 | 25.58 ± 0.71 | 74.42 ± 0.71 |
| Projection | 29.52 ± 0.11 | 96.60 | 4.15 ± 0.99 | 22.28 ± 0.07 | 113.33 ± 11.50 | 4720.90 ± 51.78 | 4.24 ± 0.41 | 95.76 ± 0.41 |
|  |  |  |  |  |  |  |  |  |
| **Efficient Profile** |  |  |  |  |  |  |  |  |
| Naive | 28.48 ± 0.13 | 91.60 | 10.62 ± 1.57 | 27.47 ± 0.07 | 1211.84 ± 21.77 | 1475.18 ± 34.24 | 75.10 ± 0.62 | 24.90 ± 0.62 |
| Adaptive (10⁻²) | 28.84 ± 0.14 | 94.20 | 7.24 ± 1.30 | 27.19 ± 0.04 | 1083.20 ± 25.06 | 1588.17 ± 30.60 | 75.76 ± 0.74 | 24.24 ± 0.74 |
| Adaptive (10⁻¹) | 28.48 ± 0.20 | 93.20 | 8.60 ± 1.42 | 24.88 ± 0.05 | 429.34 ± 28.89 | 2353.22 ± 24.97 | 87.66 ± 0.77 | 12.34 ± 0.77 |
| Adaptive (10⁰) | 27.52 ± 0.27 | 89.40 | 13.87 ± 1.80 | 23.08 ± 0.07 | 151.13 ± 14.05 | 2887.64 ± 48.09 | 94.80 ± 0.36 | 5.20 ± 0.36 |
| Fixed (10⁻²) | 27.52 ± 0.27 | 89.40 | 13.87 ± 1.80 | 23.08 ± 0.07 | 150.87 ± 13.90 | 2887.39 ± 47.88 | 94.81 ± 0.36 | 5.19 ± 0.36 |
| Fixed (10⁻¹) | 28.18 ± 0.22 | 92.00 | 10.22 ± 1.55 | 23.14 ± 0.05 | 166.68 ± 14.84 | 2874.87 ± 38.49 | 94.39 ± 0.40 | 5.61 ± 0.40 |
| Fixed (10⁰) | 28.58 ± 0.16 | 92.00 | 10.08 ± 1.53 | 27.46 ± 0.07 | 1170.99 ± 20.16 | 1501.30 ± 33.43 | 75.57 ± 0.54 | 24.43 ± 0.54 |
| Projection | 27.52 ± 0.27 | 89.40 | 13.87 ± 1.80 | 23.08 ± 0.07 | 150.87 ± 13.90 | 2887.39 ± 47.88 | 94.81 ± 0.36 | 5.19 ± 0.36 |

## Experiment Summary

| Profile | Model-Environment | Method | Experiments | Total Episodes |
|---------|-------------------|--------|-------------|----------------|
| Efficient | 2L5V 2L5V | Unsupervised | 5 | 500 |
| Efficient | 2L5V 2L5V | Projection | 5 | 500 |
| Efficient | 2L5V 2L5V | Naive | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁰) | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁻¹) | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁻²) | 5 | 500 |
| Efficient | 2L5V 2L5V | Filter_Only | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁰) | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁻¹) | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁻²) | 5 | 500 |
| Cautious | 2L5V 2L5V | Unsupervised | 5 | 500 |
| Cautious | 2L5V 2L5V | Projection | 5 | 500 |
| Cautious | 2L5V 2L5V | Naive | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁰) | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁻¹) | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁻²) | 5 | 500 |
| Cautious | 2L5V 2L5V | Filter_Only | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁰) | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁻¹) | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁻²) | 5 | 500 |

## Ablation Studies

| Profile | Model-Environment | Method | Experiments | Total Episodes |
|---------|-------------------|--------|-------------|----------------|
| Efficient | 2L5V 2L5V | Projection | 5 | 500 |
| Efficient | 2L5V 2L5V | Naive | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁰) | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁻¹) | 5 | 500 |
| Efficient | 2L5V 2L5V | Fixed (10⁻²) | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁰) | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁻¹) | 5 | 500 |
| Efficient | 2L5V 2L5V | Adaptive (10⁻²) | 5 | 500 |
| Cautious | 2L5V 2L5V | Projection | 5 | 500 |
| Cautious | 2L5V 2L5V | Naive | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁰) | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁻¹) | 5 | 500 |
| Cautious | 2L5V 2L5V | Fixed (10⁻²) | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁰) | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁻¹) | 5 | 500 |
| Cautious | 2L5V 2L5V | Adaptive (10⁻²) | 5 | 500 |
