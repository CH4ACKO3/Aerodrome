# 固定翼飞行器模板 #

在 Aerodrome 中，我们提供了一个固定翼飞行器的模板类 `#!cpp Aircraft3D`，该类继承自 `#!cpp Object3D` 类，并增加了一些用于描述固定翼飞行器特性的属性。

使用到的代码：

- Aircraft3D： `src/simulator/Core/objects/Aircraft3D.h`

## Aircraft3D ##

可以看到，`#!cpp Aircraft3D` 类增加了一些描述固定翼飞行器特性的属性，并复写了一些方法，但没有新增的方法。后面会详细解释每个方法的内容。

```cpp title="Aircraft3D.h"
class Aircraft3D : public Object3D {
public:
    double S; // 参考面积
    double c; // 特征长度

    double h; // 高度
    double q; // 动压

    double Rho; // 空气密度
    double Tem; // 温度
    double Pres; // 压力
    double a; // 声速
    double g; // 重力加速度

    double L; // 升力
    double D; // 阻力
    double N; // 侧力
    double T; // 推力
    std::array<double, 3> M; // 力矩

    Aircraft3D() {} // 默认构造函数

    Aircraft3D(py::dict input_dict) : Object3D(input_dict); // 从字典构造

    virtual void reset() override; // 重置

    virtual py::dict to_dict() override; // 属性转换为字典

    virtual py::object step(py::dict action) override; // 步进
}
```

### 构造函数 ###
`#!cpp Aircraft3D` 的构造函数接收一个参数字典 `#!cpp input_dict`，其中包含构造 `#!cpp Object3D` 类所需的参数，以及 `#!cpp Aircraft3D` 类新增的参数。

!!! info "大气属性的计算"
    构造函数和之后使用到的计算大气属性（如密度、声速等）的函数参考自公开资料，具体实现见 `src/simulator/y_atmosphere.h`。

```cpp
Aircraft3D(py::dict input_dict) : Object3D(input_dict) // 调用父类构造函数，`Object3D` 的构造函数会从 `input_dict` 中读取其需要的值
{
    V = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]); // 从传入参数的速度分量自动计算速度
    h = pos[1]; // 自动计算高度（这里假设飞行器在小范围内飞行，不考虑地面曲率，因此高度等于 y 坐标）
    S = input_dict["S"].cast<double>(); // 参考面积
    c = input_dict["c"].cast<double>(); // 特征长度

    Tem = Temperature(h); // 根据高度计算温度
    Pres = Pressure(h); // 根据高度计算压力
    Rho = Density(Tem, Pres); // 根据温度和压力计算密度
    a = SpeedofSound(Tem); // 根据温度计算声速
    g = Gravity(h); // 根据高度计算重力加速度
    
    q = 0.5 * Rho * V * V; // 计算动压

    L = 0.0; // 初始升力
    D = 0.0; // 初始阻力
    N = 0.0; // 初始侧力
    T = 0.0; // 初始推力
    M = {0.0, 0.0, 0.0}; // 初始力矩
}
```
### reset() ###
`#!cpp reset()` 函数首先调用父类 `#!cpp Object3D` 的 `#!cpp reset()` 函数，再重置 `#!cpp Aircraft3D` 类型新增的若干参数。

```cpp
virtual void reset() override
{
    Object3D::reset();
    
    h = pos[1];

    Tem = Temperature(h);
    Pres = Pressure(h);
    Rho = Density(Tem, Pres);
    a = SpeedofSound(Tem);
    g = Gravity(h);
    
    q = 0.5 * Rho * V * V;

    L = 0.0;
    D = 0.0;
    N = 0.0;
    T = 0.0;
    M = {0.0, 0.0, 0.0};
}
```

### to_dict() ###
`#!cpp to_dict()` 函数将 `#!cpp Aircraft3D` 实例的属性转换为字典并输出，方便在 Python 中使用。其首先调用父类 `#!cpp Object3D` 的 `#!cpp to_dict()` 函数，然后将 `#!cpp Aircraft3D` 类新增的属性添加到字典中。

```cpp
virtual py::dict to_dict() override // `virtual` 表示该函数可以被重写，`override` 表示重写父类中的同名函数
{
    py::dict output_dict = Object3D::to_dict(); // 调用父类 `Object3D` 的 `to_dict()` 函数
    output_dict["S"] = S; // 往 `output_dict` 中添加 `#!cpp Aircraft3D` 类新增的属性
    output_dict["c"] = c;
    output_dict["V"] = V;
    output_dict["h"] = h;
    output_dict["q"] = q;
    output_dict["Rho"] = Rho;
    output_dict["Tem"] = Tem;
    output_dict["Pres"] = Pres;
    output_dict["a"] = a;
    output_dict["g"] = g;
    output_dict["L"] = L;
    output_dict["D"] = D;
    output_dict["N"] = N;
    output_dict["T"] = T;
    output_dict["M"] = M;
    return output_dict;
}
```

### step() ###
`#!cpp step()` 是不同飞行器类中变化最大的函数。`#!cpp step()` 函数接收一个动作字典 `#!cpp action`，其中包含控制输入（如推力、舵偏角等，或过载指令，由派生类型决定）

在 `#!cpp step()` 函数中，用户需要根据动作字典 `#!cpp action` 中的控制输入更新飞行器的状态，包括但不限于计算飞行器受力、运动学步进、更新相关变量等。

在 `#!cpp Aircraft3D` 类中，飞行器所受外力（升力 `#!cpp L`、阻力 `#!cpp D`、侧力 `#!cpp N`、推力 `#!cpp T`、力矩 `#!cpp M`）以动作的形式传入，以演示基本的计算流程。

```cpp
virtual py::object step(py::dict action) override // `virtual` 表示该函数可以被重写，`override` 表示重写父类中的同名函数
{
    force_vec c_force = {T * cos(alpha) * cos(beta) - D - m * g * sin(theta),
                         T * (sin(alpha) * cos(gamma_v) + cos(alpha) * sin(beta) * sin(gamma_v)) + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta),
                         T * (sin(alpha) * sin(gamma_v) - cos(alpha) * sin(beta) * cos(gamma_v)) + L * sin(gamma_v) + N * cos(gamma_v),
                         M[0], M[1], M[2]}; // 力和力矩
    kinematics_step(c_force); // 更新状态

    h = pos[1];

    Tem = Temperature(h);
    Pres = Pressure(h);
    Rho = Density(Tem, Pres);
    a = SpeedofSound(Tem);
    g = Gravity(h);
    
    q = 0.5 * Rho * V * V;

    return to_dict();
}
```

## Space3D ##

`#!cpp Space3D` 类是 `#!cpp BaseEnv` 的子类，包含一些处理输入输出的必要属性和方法，实现比较简单。

```cpp title="Space3D.h"
class Space3D : public BaseEnv
{
private:
    double dt; // Object3D 类自身也具有 dt 参数，这里的 dt 是为了检查环境中各 Object3D 实例 dt 属性的一致性。
    double eps; // 未实装，用于检查姿态参数相容性的容差。

public:
    std::vector<std::shared_ptr<Object3D>> objects;

    Space3D(double dt, double eps) : dt(dt), eps(eps) {}

    py::object reset()
    {
        for (auto& object : objects)
        {
            object->reset();
        }
        return to_dict();
    }

    void add_object(std::shared_ptr<Object3D> object)
    {
        objects.push_back(object);
    }

    bool check_consistency()
    {
        // 相容性检查，未实装
        return true;
    }

    py::dict to_dict()
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = object->to_dict();
        }
        return output_dict;
    }

    py::object step(const py::object& actions)
    {
        py::dict result, info;

        for (auto& object : objects)
        {
            result[py::str(object->name)] = object->step(actions[py::str(object->name)]);
        }

        return result;
    }
};
```