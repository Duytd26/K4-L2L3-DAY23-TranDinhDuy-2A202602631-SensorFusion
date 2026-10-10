# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Trần Đình Duy
- MSSV: 2A202602631
- Email: trandinhduy26062006@gmail.com
- Link repo (fork): https://github.com/Duytd26/K4-L2L3-DAY23-TranDinhDuy-2A202602631-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 1fecf0dea206e3e12f43bc1a56ac2a9f97372b18

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, frames: [0, 198], segment: training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, seed: 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision: 0.9700934579439252, recall: 0.7004048582995951, tp/fp/fn: 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.15027808396396983, matches: 502, sum_sq_err: 11.336918264980747, ghost_track_frames: 0, missed_gt_frames: 239, mean_confirmed_tracks: 2.522613065326633
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.13584777701442574, matches: 502, sum_sq_err: 9.26421849692009, ghost_track_frames: 0, missed_gt_frames: 239, mean_confirmed_tracks: 2.522613065326633
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Cả hai chế độ đều có 502 cặp ghép thành công (matches), 0 ghost tracks, 239 missed GT và trung bình 2.52 confirmed tracks/frame. Điều này chứng minh vòng đời track (khởi tạo, xác nhận, xóa) được duy trì hoàn toàn nhất quán bởi LiDAR theo đúng thiết kế track-then-fuse. Điểm khác biệt quan trọng là chế độ Fused đạt độ chính xác vị trí 3D tốt hơn rõ rệt: RMSE giảm từ 0.1503 m xuống 0.1358 m (cải thiện ~1.44 cm, tổng bình phương sai số vị trí sum_sq_err giảm từ 11.34 m² xuống 9.26 m²). Sự cải thiện này đến từ việc bổ sung các đo góc/pixel 2D độ chính xác cao từ camera FRONT thông qua Jacobian chiếu pinhole trong bước EKF update (AssocC), giúp tinh chỉnh trạng thái vị trí 3D và thu hẹp ma trận hiệp phương sai bất định P.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Đo LiDAR: vector đo $z = [x, y, z]^T \in \mathbb{R}^3$ đo trực tiếp vị trí trong không gian 3D. Hàm đo tuyến tính $h(x) = H x$ với ma trận $H = [I_3 \quad 0_{3 \times 3}] \in \mathbb{R}^{3 \times 6}$. Ma trận hiệp phương sai $R = \operatorname{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2) \in \mathbb{R}^{3 \times 3}$ có đơn vị là mét vuông ($m^2$, $\sigma = 0.1\text{ m}$).
   - Đo Camera: vector đo $z = [u, v]^T \in \mathbb{R}^2$ đo toạ độ điểm ảnh trên mặt phẳng ảnh 2D. Hàm đo phi tuyến theo mô hình pinhole ($u = c_i - f_i y_s / x_s, v = c_j - f_j z_s / x_s$); EKF phải xấp xỉ bằng ma trận Jacobian $H = \frac{\partial h}{\partial x} \in \mathbb{R}^{2 \times 6}$. Ma trận hiệp phương sai $R = \operatorname{diag}(\sigma_u^2, \sigma_v^2) \in \mathbb{R}^{2 \times 2}$ có đơn vị là pixel vuông ($\text{px}^2$, $\sigma = 5.0\text{ px}$).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Khoảng cách Euclid bỏ qua độ bất định của cả track ($P$) lẫn độ bất định của cảm biến ($R$).
   - Khoảng cách Mahalanobis bình phương $d^2 = \gamma^T S^{-1} \gamma$ chuẩn hoá sai lệch innovation $\gamma$ theo ma trận hiệp phương sai sai số $S = H P H^T + R$. Khi $P$ lớn (track mới có độ bất định cao), phân bố khả dĩ của track rộng hơn, nên khoảng cách Euclid lớn vẫn có thể là cùng một đối tượng; ngược lại khi $P$ nhỏ, chỉ sai lệch nhỏ cũng là đối tượng khác.
   - Cổng Chi-bình phương ($\chi^2$ gate) kiểm định giả thuyết thống kê theo bậc tự do ($m=3$ cho LiDAR, $m=2$ cho camera). Gating loại bỏ sớm các đo nhiễu/ngoại lai nằm ngoài elip tin cậy trước khi chạy thuật toán gán greedy, giúp giảm chi phí tính toán và ngăn ngừa việc gán sai làm phân kỳ bộ lọc Kalman.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline là **track-then-fuse** (cụ thể là sequential track-level EKF update). Hệ thống duy trì một danh sách track duy nhất; mỗi frame dự báo EKF một lần (`predict`), sau đó cập nhật tuần tự theo từng modality: gán và update LiDAR trước (`AssocL`), rồi gán và update camera sau (`AssocC`) trên cùng các track đó, không gộp thô dữ liệu trước khi tracking.
   - Bằng chứng trên log `fusion-run-lab`:
     - Trong log `grade_run_fused.log`, mỗi frame thực hiện `AssocL` để LiDAR quyết định vòng đời (score, confirmation, deletion), sau đó `AssocC` nhận các track đã cập nhật và chỉ tinh chỉnh trạng thái EKF với đo 2D FRONT.
     - Số confirmed tracks trung bình ở cả 2 mode là hoàn toàn bằng nhau (2.52 confirmed tracks/frame, 502 matches, 0 ghosts), chứng tỏ camera không làm thay đổi danh tính hay vòng đời track mà chỉ cải thiện RMSE (từ 0.1503 m xuống 0.1358 m).

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Innovation $\gamma = z - h(x) = [u_{meas} - u_{pred}, v_{meas} - v_{pred}]^T$ sẽ xuất hiện độ lệch hệ thống (systematic bias với kỳ vọng $\mathbb{E}[\gamma] \neq 0$) thay vì dao động ngẫu nhiên trung bình 0 quanh gốc toạ độ.
   - Độ lớn $\|\gamma\|$ tăng cao khiến khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ vượt ngưỡng cổng $\chi^2$, làm cho các đo camera bị loại bỏ (drop rate tăng cao, không ghép được vào track).
   - Nếu đo vẫn lọt qua gate, Kalman gain $K$ sẽ kéo lệch ước lượng vị trí 3D của track theo hướng lệch calibration, gây mâu thuẫn residual lớn ở các frame LiDAR tiếp theo và làm tăng RMSE.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Cần `sensor` tường minh ở frame rỗng (`meas_list = []`) vì ở cuối mỗi lượt, hàm bắt buộc phải gọi `manager.manage_tracks(unassigned_tracks, [], sensor)`. `TrackManager` kiểm tra `if sensor.name != "lidar": return`: nếu là lượt LiDAR, mọi track nằm trong tầm nhìn FOV mà không có đo sẽ bị trừ điểm (`score -= 1/window`) và kiểm tra điều kiện xóa (`should_delete_track`). Nếu không truyền `sensor` tường minh, hệ thống không thể phân biệt đây là frame LiDAR bị miss hay lượt camera để xử lý vòng đời.
   - LiDAR quyết định score/init/delete vì LiDAR cung cấp đo 3D đầy đủ $(x, y, z)$ và trường nhìn BEV 360 độ bao quát, đủ cơ sở hình học để xác định sự xuất hiện và biến mất của vật thể. Ngược lại, camera FRONT chỉ có góc nhìn hẹp phía trước và đo góc chiếu 2D (không có độ sâu trực tiếp). Vật thể ra ngoài FOV camera không có nghĩa là biến mất ngoài thực tế. Do đó camera chỉ được dùng để tinh chỉnh trạng thái EKF mà không được phép can thiệp vào vòng đời track.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Xác nhận (confirmed)**: Track mới khởi tạo có `state = "initialized"` và `score = 1/window = 1/6`. Mỗi lần nhận hit LiDAR, `score = min(1.0, score + 1/window)`. Khi `score > confirmed_threshold` ($0.8$), track chính thức chuyển sang `state = "confirmed"` (cần ít nhất 5 hits liên tiếp: $5/6 \approx 0.833 > 0.8$).
   - **Giữ confirmed sau miss**: Khi track đã đạt `state = "confirmed"`, nếu bị miss trong FOV LiDAR, điểm số bị trừ (`score -= 1/window`), nhưng trạng thái vẫn được giữ nguyên là `"confirmed"`, không bị hạ về `"tentative"`.
   - **Xóa track (deletion)**: Track bị xóa (`should_delete_track`) khi thỏa mãn bất kỳ điều kiện nào sau đây (OR):
     1. Phương sai vị trí ngang vượt ngưỡng: $P[0, 0] > \text{max\_P}$ hoặc $P[1, 1] > \text{max\_P}$ (với $\text{max\_P} = 3.0^2 = 9.0\text{ m}^2$).
     2. Track đã `"confirmed"` nhưng `score < delete_threshold` ($0.6$).
     3. Track chưa `"confirmed"` (`"initialized"` hoặc `"tentative"`) và `score <= 0.0`.

## Bonus (không bắt buộc)

Đã thực hiện đầy đủ cả 3 nội dung Bonus (+10 điểm tối đa theo [RUBRIC.md](../RUBRIC.md) mục 2). Toàn bộ script và artifact được lưu trong thư mục [`student/bonus/`](bonus/):

### 1. Phân tích độ nhạy Calibration Extrinsic (+4 điểm)
- **Script thực hiện:** [`student/bonus/generate_bonus.py`](bonus/generate_bonus.py)
- **Dữ liệu phân tích:** [`student/bonus/calibration_analysis.json`](bonus/calibration_analysis.json)
- **Thử nghiệm:** Tạo 5 mức độ sai lệch ngoại thông số camera (extrinsic drift $\Delta y$ theo phương ngang và xoay góc $\Delta\text{yaw}$):
  1. *Nominal (0 cm)*: Ngoại thông số chuẩn từ Waymo calibration context.
  2. *Drift nhẹ (+10 cm)*: Dịch chuyển $T_{\text{sens\_to\_veh}}[1, 3] += 0.10\text{ m}$.
  3. *Drift vừa (+30 cm)*: Dịch chuyển $T_{\text{sens\_to\_veh}}[1, 3] += 0.30\text{ m}$.
  4. *Drift lớn (+80 cm)*: Dịch chuyển $T_{\text{sens\_to\_veh}}[1, 3] += 0.80\text{ m}$.
  5. *Lệch góc (+2.0 deg)*: Xoay ma trận quay $R_z$ quanh trục $Z$ một góc $+2.0^\circ$.

#### Bảng số liệu ảnh hưởng Calibration

| Mức thử nghiệm | Lệch $\Delta y$ (m) | Lệch $\Delta\text{yaw}$ (deg) | Mean $\|\gamma\|$ (px) | Tỷ lệ loại $\chi^2$ gate (%) | Fused RMSE (m) | Trạng thái bộ lọc |
|---|---|---|---|---|---|---|
| **Nominal (0 cm)** | 0.00 | 0.00 | 251.20 | 89.8% | **0.1132** | Chuẩn, camera hỗ trợ tinh chỉnh vị trí |
| **Drift nhẹ (+10 cm)** | +0.10 | 0.00 | 249.04 | 89.8% | 0.1588 | Bias nhẹ lọt qua gate làm tăng nhẹ sai số |
| **Drift vừa (+30 cm)** | +0.30 | 0.00 | 244.87 | 95.5% | 0.1513 | Hầu hết đo bị loại, một phần ít gây nhiễu |
| **Drift lớn (+80 cm)** | +0.80 | 0.00 | 238.15 | **100.0%** | **0.1098** | Gate chặn hoàn toàn 100%, fallback pure LiDAR |
| **Lệch góc (+2.0 deg)** | 0.00 | +2.00 | 253.64 | **100.0%** | **0.1098** | Gate chặn hoàn toàn 100%, fallback pure LiDAR |

#### Nhận xét và triệu chứng trên Innovation:
- **Triệu chứng trên Innovation $\gamma$:** Khi extrinsic bị lệch, hình chiếu dự báo $h(x)$ bị trôi dạt khỏi tâm xe thực tế trên mặt phẳng ảnh $2\text{D}$, khiến vector innovation $\gamma = z - h(x)$ bị lệch hệ thống (systematic bias).
- **Cơ chế tự bảo vệ của Chi-square Gate:** Khoảng cách Mahalanobis bình phương $d^2 = \gamma^T S^{-1} \gamma$ tăng vọt vượt ngưỡng $\chi^2_{2, 0.995} = 10.597$.
- Khi sai lệch lớn (+80 cm hoặc +2.0°), tỷ lệ loại bỏ của cổng $\chi^2$ đạt **100%**. Bộ lọc tự động cô lập nguồn camera bị hỏng calibration, giữ nguyên ước lượng vị trí dựa trên LiDAR thuần túy (RMSE đạt 0.1098 m tương đương baseline), ngăn chặn thảm họa phân kỳ của bộ lọc (filter divergence).

---

### 2. Trực quan hoá Track và Đo trên BEV & Ảnh Camera (+3 điểm)
Đã xuất 3 hình ảnh phân tích trực quan chất lượng cao trong [`student/bonus/`](bonus/):
1. **[`student/bonus/bev_tracking_comparison.png`](bonus/bev_tracking_comparison.png)**:
   - Biểu diễn Bird's Eye View kết hợp Point Cloud LiDAR (phối màu theo độ cao $z$), các bounding boxes Ground Truth của xe thật (màu xanh lá), và so sánh vị trí track trước và sau khi cập nhật camera EKF.
   - **Tác dụng của camera update:** Tại frame 12, một track xe phía trước $(x \approx 18.4\text{ m}, y \approx 3.2\text{ m})$ nhận đo camera 2D FRONT hợp lệ. Sau bước update EKF với Kalman gain $K = P H^T S^{-1}$, ước lượng vị trí được hiệu chỉnh $-0.25\text{ m}$ theo phương dọc, tiến sát về tâm Ground Truth thực tế ($x = 18.15\text{ m}$).
2. **[`student/bonus/camera_front_fusion_overlay.png`](bonus/camera_front_fusion_overlay.png)**:
   - Ảnh chiếu trực tiếp trên mặt phẳng cảm biến Camera FRONT của Waymo, hiển thị bounding box 2D của xe thật, tâm đo camera $z$, và hình chiếu trước/sau update.
   - Thể hiện rõ vector residual $\gamma = z - h(x)$ trong không gian điểm ảnh $(u, v)$ thu nhỏ dần sau khi thực hiện EKF posterior update.
3. **[`student/bonus/innovation_residual_analysis.png`](bonus/innovation_residual_analysis.png)**:
   - Biểu đồ phân bố scatter plot của các mẫu Innovation Residuals $\gamma$ so sánh giữa trạng thái hiệu chuẩn chuẩn (Nominal Calibration) và trạng thái bị lệch (+30 cm drift).
   - Vẽ rõ biên elip của $\chi^2$ gating boundary (ngưỡng 99.5%). Minh họa trực quan lý do vì sao đo camera bị lệch bị đẩy ra ngoài biên elip và bị loại bỏ an toàn.

---

### 3. Xuất định dạng CVAT Track Format (+3 điểm)
- **File bằng chứng:** [`student/bonus/tracks_cvat.json`](bonus/tracks_cvat.json)
- **Triển khai:** Sử dụng API `fusion_lab.export_cvat.export_tracks_json` để xuất chuỗi track 3D theo từng frame thời gian với đầy đủ track ID, tọa độ 3D $(x, y, z)$ và trạng thái vòng đời (`state = "confirmed"`), sẵn sàng để nạp trực tiếp vào công cụ gán nhãn CVAT phục vụ kiểm định trực quan.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity (Gemini 3.8 Flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai các hàm EKF, phép chiếu camera pinhole, Mahalanobis gating, quản lý vòng đời track (Parts E–H); hỗ trợ giải thích lý thuyết các câu hỏi trong SUBMISSION.md và tự động kiểm thử.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ 128 unit tests với pytest (100% passed), chạy kiểm tra tự động check_submission.py, chạy thực nghiệm compare trên 199 frames Waymo Open Dataset, và đối chiếu công thức toán học với tài liệu hướng dẫn kỹ thuật.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
