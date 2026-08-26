# Multi-Instrument Atmospheric Latent Fusion + Swin Forecast

这个目录是直接接在你当前 `sat_test` 分支的 `satellite/` 自编码器之后的第二阶段。

当前 `BAMUAAutoEncoder` 已经提供：

```text
irregular 1BAMUA observations
        -> PointEncoder
        -> SetConvOffToOn
        -> latent [B,D,H,W] + density [B,1,H,W]
        -> SetConvOnToOff + PointDecoder
        -> BT observations
```

本项目增加：

```text
z_1bamua --Adapter--\
z_mhs   --Adapter----> confidence-aware Fusion
z_atms  --Adapter---/          |
                              v
                    Common Atmosphere Z_t
                      [B,C,L,H,W]
                              |
                              v
                     3-D Swin Transformer
                              |
                              v
                    Common Atmosphere Z_t+6h
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
              Head_A       Head_B       Head_CONV
                 |            |            |
                 v            v            v
             z_A(t+6)      z_B(t+6)     z_conv(t+6)
                 |
           frozen AE decoder
                 |
                 v
          observation forecast
```

## 为什么先预计算每个仪器 latent

训练 Fusion/Swin 时不再反向传播单仪器 AE。每个 AE 是已经学好的“仪器语言”，先冻结并把每个 6 小时时间片编码成一个小得多的 Zarr：

```text
time        [T]
latent      [T,D,H,W]
density     [T,1,H,W]
available   [T]
latent_mean [D]
latent_std  [D]
```

这样训练天气预报时不需要每个 step 都重新读取几十万条原始卫星观测并跑 SetConv。

## 1. 放到现有仓库

把整个 `atmosphere/` 文件夹复制到：

```text
C:/Users/Lenovo/code/local_code/aardvark-weather-public/atmosphere/
```

仓库结构会变为：

```text
aardvark-weather-public/
├── satellite/          # 你已经完成的单仪器 AE
├── atmosphere/         # 本项目
├── ...
```

安装附加依赖：

```bash
python -m pip install -r requirements_atmosphere.txt
```

## 2. 先把 1BAMUA AE 输出预计算成 latent Zarr

修改：

```text
atmosphere/configs/precompute_bamua.yaml
```

其中最重要的是训练好的 checkpoint：

```yaml
checkpoint: "runs/.../best.pth"
```

然后在仓库根目录：

```bash
python -m atmosphere.precompute_instrument_latents --config atmosphere/configs/precompute_bamua.yaml
```

这个脚本直接兼容你 `sat_test` 分支现在的：

```python
satellite.models.BAMUAAutoEncoder
satellite.config.BAMUAConfig
satellite.datasets.BAMUAZarrDataset
BAMUAAutoEncoder.encode_chunked(...)
```

它使用完整 6h observation set，但通过 `encode_chunked` 分块，因此不会把约 40 万点一次性塞进 SetConv。

## 3. 每个新仪器都做同样的事

前提是每个仪器 AE 统一提供：

```python
encode_chunked(...) -> latent, density
```

并且自己的 Dataset 提供：

```python
get_full_sample(i)
```

然后复制一个 precompute YAML，只改：

```yaml
instrument:
raw_zarr:
checkpoint:
output_zarr:
model_class:
config_class:
dataset_class:
```

注意：不同仪器的 `latent_dim D` 可以不同，但 **H/W 必须相同**。这正是所有独立 AE 从一开始应该遵守的公共接口。

## 4. Fusion 的数学含义

每个仪器先通过自己的 Adapter：

```math
z_t^k \in R^{D_k\times H\times W}
\rightarrow
u_t^k \in R^{C\times L\times H\times W}
```

这里 `L` 是共同大气隐空间的垂直维。

如果 `pressure_levels_hpa=null`，L 是 learned latent levels；如果提供气压数组，就给共同状态加入显式 pressure encoding。

不同仪器不会直接 channel concat。Fusion 在每个 `(level,lat,lon)` 位置计算仪器权重：

```math
\alpha_k = softmax_k(score(u_k, density_k))
```

再得到：

```math
Z_t = \sum_k \alpha_k u_t^k
```

所以以后某个时间缺 ATMS 或 CrIS，不需要改变网络输入维度。

## 5. Swin 才是真正做“预报”的部分

```math
Z_t -> P_Swin -> Z_{t+6h}
```

Swin 工作在：

```text
vertical latent level × latitude × longitude
```

上，使用 3-D shifted-window attention；经度按周期边界处理。

Processor 最后预测 residual：

```math
Z_{t+6h} = Z_t + delta_Z
```

`delta_head` 初始化为 0，因此训练一开始接近 persistence，比直接随机预测未来共同状态稳定。

## 6. 从共同状态还原回每个仪器 latent

每个仪器有一个很小的：

```text
AtmosphereToInstrumentHead_k
```

做：

```math
Z_t -> z_t^k
```

同时预测 `log(1+density)`。

训练包含两部分：

```math
L = lambda_rec L_reconstruct(t)
  + lambda_forecast L_forecast(t+6h,...)
  + lambda_density L_density
```

其中：

- `L_reconstruct(t)`：共同状态必须还能解释当前各仪器 latent；
- `L_forecast`：Swin 产生的未来共同状态必须能解释未来各仪器 latent；
- latent loss 只在 target AE 的 `density > threshold` 区域计算，避免无观测区域把 loss 淹没。

## 7. 先做模型纯张量 smoke test

不需要 Zarr：

```bash
python -m atmosphere.tests.smoke_model
```

它会验证：

```text
2 个假仪器
-> fusion
-> [B,C,L,H,W]
-> Swin
-> instrument heads
-> loss.backward()
```

## 8. 用真实 latent 做 smoke training

先修改：

```text
atmosphere/configs/fusion_smoke.yaml
```

然后：

```bash
python -m atmosphere.train_fusion --config atmosphere/configs/fusion_smoke.yaml
```

目前只有 1BAMUA 也能跑，但那时 Fusion 退化成单模态。等第二个仪器的 latent store 加进 YAML 后，才真正开始学习跨仪器共同大气状态。

## 9. 正式训练

```bash
python -m atmosphere.train_fusion --config atmosphere/configs/fusion_train.yaml
```

checkpoint：

```text
runs/atmosphere_fusion/best.pth
runs/atmosphere_fusion/last.pth
```

继续训练：

```bash
python -m atmosphere.train_fusion \
  --config atmosphere/configs/fusion_train.yaml \
  --resume runs/atmosphere_fusion/last.pth
```

Windows PowerShell 可写成一行。

## 10. 测试

```bash
python -m atmosphere.evaluate_fusion \
  --config atmosphere/configs/fusion_train.yaml \
  --checkpoint runs/atmosphere_fusion/best.pth \
  --output runs/atmosphere_fusion/test_metrics.json
```

指标按：

```text
instrument / lead step
```

输出 latent-space MSE。

## 11. 生成未来 instrument latent

例如向前滚 4 个 6h，也就是 24h：

```bash
python -m atmosphere.forecast_latents \
  --config atmosphere/configs/fusion_train.yaml \
  --checkpoint runs/atmosphere_fusion/best.pth \
  --sample-index 0 \
  --steps 4 \
  --output runs/atmosphere_fusion/forecast_24h.pth
```

输出中每一个 lead time 都有：

```python
result["steps"][lead]["instruments"][name]["latent"]
result["steps"][lead]["instruments"][name]["density"]
```

它们已经被反归一化回该仪器 AE 原本的 latent 尺度，可以接回冻结的 instrument AE decoder。

例如 1BAMUA 后续可以：

```python
bt_pred = bamua_ae.decode(
    latent=predicted_latent.unsqueeze(0).cuda(),
    density=predicted_density.unsqueeze(0).cuda(),
    lon=query_lon,
    lat=query_lat,
    satellite_id=query_satellite_id,
    is_land=query_is_land,
    sample_time=future_time,
    obs_time=query_obs_time,
    zenith=query_zenith,
    azimuth=query_azimuth,
)
```

这里未来 BT 的 query metadata 必须由实际未来卫星轨道/扫描几何提供；Decoder 不能凭空决定未来观测位置和扫描角。

## 12. 常规观测怎么接进来

常规观测也训练自己的 AE，只要最终输出相同 H/W 的：

```text
z_conv [D_conv,H,W]
```

就把：

```yaml
conv: "F:/.../conventional_latent.zarr"
```

加入 `data.instruments`。名字虽然叫 instruments，代码实际上把它当 modality，所以常规观测也完全可以放进去。

未来：

```text
Z_atmosphere(t+lead)
-> Head_CONV
-> z_conv(t+lead)
-> frozen conventional decoder
-> regular global gridded field
```

这正对应你的目标：卫星和常规观测共同形成一个大气隐空间，但最终仍可以分别回到各自观测空间。

## 13. 新仪器增量接入

这版先实现完整 joint training。代码结构已经把每个仪器隔离成：

```text
Adapter_k + InstrumentHead_k
```

所以后续做你提出的“新仪器锚点更新”时，可以冻结：

```text
旧 Adapter
旧 Head
Fusion backbone
Swin processor
```

只新增/训练：

```text
Adapter_new
Head_new
IncrementalFusionUpdate_new
```

这一步建议在当前 joint fusion + forecast baseline 跑通后再加，否则很难判断问题来自基础 forecast 还是增量接入机制。
