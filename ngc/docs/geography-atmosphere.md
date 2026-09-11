# 地理位置、高度与空气物理量

新增 `core.geodesy`、`core.lookup` 和 `models.atmosphere`。数值核为 JAX 纯函数，支持 JIT、vmap、scan 和有效域内自动微分；地理与气象表是显式参数 PyTree，可放入 `WorldParameters.resources`。文件读取、网格预处理和来源管理留在宿主端。模块不查询网络天气、不隐式下载 DEM/大地水准面资产。

## 坐标、高度和方向

`lla = [latitude_rad, longitude_rad, ellipsoid_height_m]`，纬度为大地纬度，椭球采用 WGS84。推荐 float64：ECEF 数值约数百万米，float32 不适合精细定位。

| 接口 | 用途 |
|---|---|
| `geodetic_to_ecef` / `ecef_to_geodetic` | 经纬椭球高 ↔ 地心地固直角坐标 |
| `geodetic_to_ned` / `ned_to_geodetic` | 指定地理参考点的局部 NED 位移 ↔ 经纬高 |
| `geocentric_radius(lat,h)` | 在给定大地纬度和椭球高处的地心距离 |
| `radius_to_ellipsoid_height(r,lat)` | 已知大地纬度，将地心距离还原为椭球高 |
| `direction_offset` | NED 位移变换为相对某水平航向的前/右/下分量 |
| `relative_position` | 两地理点的相对分量、右正方位偏角、上正仰角、斜距 |

方向几何采用 ECEF 差向量在原点局部 NED 的投影，适合局部制导、传感器视线和场景布置。斜距不是地表距离，尚未实现全球椭球测地线、沿地表航线或磁航向。经度输出在 ±π 区间；跨日期变更线坐标计算连续，角度输出有 wrap 分支。极点经度约定为零，极点不是可微的经纬坐标图。逆变换使用固定 10 次迭代，限制椭球高不低于 -1 km；已测试至 10000 km 高度，不用于地球内部。

```python
from aerodrome.core import geodesy as geo
origin = jnp.array([jnp.deg2rad(22.5), jnp.deg2rad(114.), 3030.])
target = geo.ned_to_geodetic(jnp.array([1000., 200., -100.]), origin)
offset, azimuth_error, elevation, slant_range = geo.relative_position(
    target, origin, heading_rad=jnp.deg2rad(15.))
```

高度定义必须显式：

| 量 | 转换与约定 |
|---|---|
| 椭球高 h | GNSS/LLA 几何位置使用的高度 |
| 正高 H，近似海拔 | `H = h - N`；N 为匹配基准的大地水准面起伏 |
| 离地高 AGL | `H - terrain_H`；两者必须使用同一垂直基准 |
| 标准位势高度 | `r0*z/(r0+z)`，r0=6356766 m，仅标准大气球形近似 |
| 地心距离 r | WGS84 几何计算；取决于纬度，不能统一减去一个地球半径 |

`ellipsoid_to_orthometric`、`orthometric_to_ellipsoid`、`height_above_ground`、`geometric_to_geopotential` 和反向函数提供这些转换。标准位势高度转换不是真实地球重力势求解；正高也不是所有地区高程基准的通用替代。

## 显式查表

`regular_grid(axes,values)` 在宿主端检查 1～4 维有限、严格递增坐标轴和数值形状；`lookup(grid,point)` 为多线性插值，支持尾部通道维度，网格外返回 NaN，不自动外推。

```python
from aerodrome.core.lookup import regular_grid, lookup
# 示例 N=30 m；实际应导入匹配基准的数据，不代表当地真实水准面。
geoid = regular_grid(([.3,.5], [1.9,2.1]), [[30.,30.],[30.,30.]])
H = geo.height_from_geoid_grid(lla, geoid)

# 指定纬度，生成高度—地心距离表，也可反向查表。
heights = jnp.linspace(0., 10000., 101)
radii = jax.vmap(lambda h: geo.geocentric_radius(.6,h))(heights)
height_table = regular_grid((radii,), heights)
height = lookup(height_table, (radius_m,))
```

查表为近似，能直接用解析转换时优先解析函数。表的轴顺序、单位、水平/垂直基准、日期和来源应写入既有 Asset manifest。经度周期接缝不自动处理：可先将查询经度映射到表区间，并在数据预处理中补齐一致的周期端点。NaN 缺测值必须由明确的预处理规则解决；构造器不会偷偷填补。地形、大地水准面和天气数据不内置，示例使用明确标注的合成数据。

## 三种空气物理量入口

1. `standard_temperature_pressure(z)`：US1976 分层低层大气，输入几何高度，内部先转位势高度。范围 **-1～80 km**；负高度是首层递减率的延伸。返回温度 K、静压 Pa。`pressure_to_standard_height(P)` 是对应反解，得到标准气压高度，不等于实际海拔。
2. `air_properties(T,P,RH)`：实测温压湿度计算密度、声速、动力/运动黏度。RH 为 0～1，不是百分数。`valid` 描述热力输入是否在支持范围；风速和重力有各自的输入范围，不能只靠此标志检查整个环境。
3. `atmosphere_from_grid(lla,weather_grid,geoid_grid,time_s=...)`：天气表坐标为 `[lat,lon,H,(time)]`，六个通道为 `[T_K,P_Pa,RH,wind_N,wind_E,wind_D]`。支持 3D 静态或 4D 随时间变化的场。当前所有通道线性插值，包括气压；这不是重新求解静力平衡。

便捷接口 `atmosphere_at_lla` 将 `h-N` 作为标准大气几何高度近似，要求显式传入 N（可以显式选择零近似），还支持温度偏差、压力覆盖值、湿度和 NED 风速。改变温度偏差不会重新积分整根空气柱；如果需要真实压力，请提供测量值或气象表。经纬度本身不会自动生成天气，纬度只参与近似重力计算。

```python
from aerodrome.models import atmosphere as atm
sample = atm.atmosphere_at_lla(
    lla, geoid_undulation_m=30., temperature_offset_K=10.,
    relative_humidity=.4,
    wind_ned_m_s=atm.meteorological_wind(12., jnp.deg2rad(270.)))
airdata, mach, reynolds, qbar = atm.flow_conditions(
    sample, velocity_body_m_s, rotation_nb, reference_length_m=3.)
```

气象风向是“从哪来”，自北顺时针；上例西风变成向东的空气团速度。`flow_conditions` 扣除风速后计算真空速、气动角、Mach、Reynolds 和动压。参考长度须为正。

## 物理范围

- 干空气密度使用理想气体，支持 150～350 K；湿空气为干空气/水蒸气理想混合物。
- 饱和蒸气压采用液水 Tetens 近似，此版非零湿度仅支持 **0～50°C**。低温干空气受支持；低温湿空气/冰面饱和、凝结、云液水和降水未实现。
- 湿空气声速采用恒定组分比热近似。黏度采用干空气 Sutherland 近似，不是完整湿空气输运模型。
- 重力使用 Somigliana 表面值加球形反平方高度修正，非真实重力场，也未提供地球自转、科氏项或重力异常。
- 无效温压湿度、越界大气高度、越界查表均返回 NaN/无效标志，不静默夹到边界；分层/网格单元接缝的一阶导数可能变化。
- 新环境模型尚未替换 F-16 来源复现实验的原有大气拟合。可以通过刚体 `load_fn` 和共享 resources 显式选用，避免静默改变既有实验结果。

## 重现与依据

执行 `python examples/geography_atmosphere.py`，生成 `artifacts/geography_atmosphere/` 下的标准大气 CSV、北纬 22.5° 高度—地心距离 CSV、NPZ 和场景摘要。两张 CSV 分开标注几何高度和椭球高。

测试覆盖 WGS84 特殊点、随机坐标往返、极点/日期变更线、局部方向、解析高度与查表、SciPy 网格对照、标准层参考值/连续性/反解、湿空气、风速、气象四维插值、精度保持与梯度。尚未使用外部全球 DEM/EGM/气象产品进行实测验证。

参考：[NGA WGS84](https://earth-info.nga.mil/?action=wgs84&dir=wgs84)、[NOAA/NGS 高度基准说明](https://www.ngs.noaa.gov/GEOID/geoid_def.html)、[US Standard Atmosphere 1976](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539.pdf)、[NWS 蒸气压计算](https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf)、[NASA 黏度模型](https://www.grc.nasa.gov/www/winddocs/user/keywords/viscosity.html)。
