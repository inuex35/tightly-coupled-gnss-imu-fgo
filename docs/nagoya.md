# Nagoya generalization

The tokyo-tuned defaults carried to the nagoya drives of
[PPC-Dataset](https://github.com/taroz/PPC-Dataset) unchanged — same
configuration, only the lever arm follows the dataset README (FRD
`0.593,-0.670,-1.216` converted to the internal FLU convention).

![nagoya_defaults](nagoya_defaults.png)

| run  | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|---------|--------|--------|
| run1 |  7602 ep  | 34.80 m | 0.41 m  | 0.23 m  | 45.2 % | 53.6 % |
| run2 |  9451 ep  | 27.26 m | 0.56 m  | 0.27 m  | 48.1 % | 49.9 % |
| run3 |  5201 ep  | 15.99 m | 2.11 m  | 0.96 m  | 29.6 % | 28.9 % |

The nagoya routes run under elevated expressways with repeated full
GNSS outages: the AllRMS is dominated by IMU-only stretches, while
the fix quality transplants (FixRMS 0.23–0.96 m).

Run:

```bash
LEVER_ARM=0.593,0.670,1.216 \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```
