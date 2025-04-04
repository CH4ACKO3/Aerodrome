# 固定翼飞行器模板 #

在 Aerodrome 中，我们提供了一个固定翼飞行器的模板类 `Aircraft3D`，该类继承自 `Object3D` 类，并增加了一些用于描述固定翼飞行器特性的属性。

使用到的代码：

- Aircraft3D： `src/simulator/Core/objects/Aircraft3D.h`

## Aircraft3D ##

可以看到，`Aircraft3D` 类增加了一些描述固定翼飞行器特性的属性，并复写了一些方法，但没有新增的方法。后面会详细解释每个方法的内容。

```cpp
class Aircraft3D : public Object3D {
public:
    double S; // 参考面积
    double c; // 特征长度
    double m; // 质量

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

    virtual Object3D d() override; // 运动学导数
}
```

### 构造函数 ###
`Aircraft3D` 的构造函数接收一个参数字典 `input_dict`，其中包含构造 `Object3D` 类所需的参数，以及 `Aircraft3D` 类新增的参数。

> **Note**  
> 构造函数和之后使用到的计算大气属性（如密度、声速等）的函数参考自公开资料，实现可见 `src/simulator/y_atmosphere.h`。

```cpp
Aircraft3D(py::dict input_dict) : Object3D(input_dict) // 调用父类构造函数，`Object3D` 的构造函数会从 `input_dict` 中读取其需要的值
{
    V = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]); // 从传入参数的速度分量自动计算速度
    h = pos[1]; // 自动计算高度（这里假设飞行器在小范围内飞行，不考虑地面曲率，因此高度等于 y 坐标）
    S = input_dict["S"].cast<double>(); // 参考面积
    c = input_dict["c"].cast<double>(); // 特征长度
    m = input_dict["m"].cast<double>(); // 质量（也可以定义为初始质量）

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

### to_dict() ###
`to_dict()` 函数将 `Aircraft3D` 实例的属性转换为字典并输出，方便在 Python 中使用。其首先调用父类 `Object3D` 的 `to_dict()` 函数，然后将 `Aircraft3D` 类新增的属性添加到字典中。

```cpp
virtual py::dict to_dict() override // `virtual` 表示该函数可以被重写，`override` 表示重写父类中的同名函数
{
    py::dict output_dict = Object3D::to_dict(); // 调用父类 `Object3D` 的 `to_dict()` 函数
    output_dict["S"] = S; // 往 `output_dict` 中添加 `Aircraft3D` 类新增的属性
    output_dict["c"] = c;
    output_dict["m"] = m;
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

### d() ###
`d()` 函数计算 `Aircraft3D` 实例的运动学导数。假设不考虑加速度和角加速度的变化（即，加速度和角加速度是由飞行器所受力引起或由输入决定的），则求解以下几个量的导数可以满足计算要求：

- 位置（`pos`）
- 速度（包括速度大小和角度，即`V`、`theta_v`、`phi_v`）
- 姿态角（`theta`、`phi`、`gamma`）
- 角速度（`ang_vel`）

> **Note**  
> 描述飞行器的位置姿态还有其它参数，例如迎角 `alpha`、侧滑角 `beta` 等；还有一些量需要随着步进更新，例如飞行器的高度 `h`、动压 `q` 等。但这些量并不是相互独立的，
> 因此可以只求解上述几个量，在 `step()` 函数中完成运动学积分之后，再根据需要更新其它量。

返回的是和调用该方法的实例类型相同的一个新实例，其每个属性的值是原实例对应属性的导数。例如，返回实例的 `V` 属性是传入实例的 `V` 属性的导数，即加速度（更准确地说是 `V` 的变化率）
```cpp
virtual Object3D d() override
{   
    auto derivative = *this;
    
    derivative.V = (T * cos(alpha) * cos(beta) - D - m * g * sin(theta_v)) / m;
    derivative.theta_v = (T * (sin(alpha) * cos(gamma_v) - cos(alpha) * sin(beta) * sin(gamma_v))
                            + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta_v)) / (m * V);
    derivative.phi_v = -(T * (sin(alpha) * sin(gamma_v) - cos(alpha) * sin(beta) * cos(gamma_v))
                        + L * sin(gamma_v) + N * cos(gamma_v)) / (m * V * cos(theta_v));

    derivative.ang_vel[0] = (M[0] - (J[2] - J[1]) * ang_vel[1] * ang_vel[2]) / J[0];
    derivative.ang_vel[1] = (M[1] - (J[0] - J[2]) * ang_vel[2] * ang_vel[0]) / J[1];
    derivative.ang_vel[2] = (M[2] - (J[1] - J[0]) * ang_vel[0] * ang_vel[1]) / J[2];

    derivative.theta = ang_vel[1] * sin(gamma) + ang_vel[2] * cos(gamma);
    derivative.phi = (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma)) / cos(theta);
    derivative.gamma = ang_vel[0] * - tan(theta) * (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma));

    derivative.pos[0] = V * cos(theta_v) * cos(phi_v);
    derivative.pos[1] = V * sin(theta_v);
    derivative.pos[2] = -V * cos(theta_v) * sin(phi_v);

    return derivative;
}
```

### step() ###
`step()` 是不同飞行器类中变化最大的函数。`step()` 函数接收一个动作字典 `action`，其中包含控制输入（如推力、舵偏角等，或过载指令，由派生类型决定）和时间步长 `dt`。

> **Note**  
> 由于积分步长 `dt` 在环境运行过程中有可能会变化（以支持一些自适应步长的算法），因此 `dt` 并没有硬编码在 `Aircraft3D` 类的属性中，而是随动作传入。

在 `step()` 函数中，用户需要根据动作字典 `action` 中的控制输入更新飞行器的状态，包括但不限于计算飞行器受力、运动学步进、更新相关变量等。

在 `Aircraft3D` 类中，飞行器所受外力（升力 `L`、阻力 `D`、侧力 `N`、推力 `T`、力矩 `M`）以动作的形式传入，以演示基本的计算流程。

> **Note**  
> 由于在 `Object3D` 类中[重载](https://zh.cppreference.com/w/cpp/language/operators)了四则运算符，将两个 `Object3D` 或 `Object3D` 的子类之间的加减乘除定义为其某些属性的对应运算（具体实现见下面代码），在动力学积分时可以更方便地处理数值计算（例如涉及微分的运算）。

```cpp
virtual py::object step(py::dict action) override // `virtual` 表示该函数可以被重写，`override` 表示重写父类中的同名函数
{
    double dt = action["dt"].cast<double>(); // 从动作字典中读取积分步长
    
    L = action["L"].cast<double>(); // 从动作字典中读取飞行器所受外力
    D = action["D"].cast<double>();
    N = action["N"].cast<double>();
    T = action["T"].cast<double>();
    M = action["M"].cast<std::array<double, 3>>();

    // 使用不同的积分方法更新飞行器状态

    if (integrator == "euler")
    {
        // 这里可以直接运算是因为在 Object3D 类中重载了四则运算
        // this->d() 调用求导方法，返回动力学导数（即，返回的实例的每个属性是传入的实例对应属性的导数）
        // this->d() * dt 即表示增量
        *this = *this + this->d() * dt;
    }
    else if (integrator == "midpoint")
    {
        auto temp1 = *this + this->d() * (0.5 * dt);
        auto k1 = temp1.d();
        *this = *this + k1 * dt;
    }
    else if (integrator == "rk23")
    {
        auto k1 = this->d();
        auto temp1 = *this + k1 * (0.5 * dt);
        auto k2 = temp1.d();
        auto temp2 = *this + k2 * (0.5 * dt);
        auto k3 = temp2.d();
        *this = *this + (k1 + k2 * 2 + k3) * (dt / 4);
    }
    else if (integrator == "rk45")
    {
        auto k1 = this->d();
        auto temp1 = *this + k1 * (0.5 * dt);
        auto k2 = temp1.d();
        auto temp2 = *this + k2 * (0.5 * dt);
        auto k3 = temp2.d();
        auto temp3 = *this + k3 * dt;
        auto k4 = temp3.d();
        *this = *this + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6);
    }
    
    // 由于在上面的动力学积分中只计算了必要的量
    // 此处还需要对未更新的属性进行计算
    beta = cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma);
    alpha = (cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta);
    gamma_v = (cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v);

    vel[0] = V * cos(theta_v) * cos(phi_v);
    vel[1] = V * sin(theta_v);
    vel[2] = -V * cos(theta_v) * sin(phi_v);
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

`Space3D` 类是 `BaseEnv` 的子类，包含一些处理输入输出的必要属性和方法，实现比较简单。

```cpp
class Space3D : public BaseEnv
{
private: // [tl! collapse:start]
    double tau; // 每次积分的时间长度，tau = dt / integrate_steps
    double dt; // 每次被调用 step() 方法时，步进的时间长度
    double eps; // 容差，可以用于检查动力学积分的精度，未实装
    int integrate_steps; // 每次被调用 step() 方法时，调用环境内对象的 step() 方法的次数（将每次步进划分为几次积分） // [tl! collapse:end]

public:
    std::vector<std::shared_ptr<Object3D>> objects;

    Space3D(double tau, double eps, int integrate_steps)
        : tau(tau), eps(eps), integrate_steps(integrate_steps)
    {
        dt = tau / integrate_steps;
    }

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
        objects.push_back(object); // 将传入的对象(Object3D及其子类)存入列表中
    }

    bool check_consistency()
    {
        return true;
    }

    py::dict to_dict() // 调用环境中每个对象的 to_dict() 方法，并将所有返回值存入一个字典中并返回
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = object->to_dict(); 
        }
        return output_dict;
    }

    py::dict get_d() // 获取环境中所有对象的运动学导数，用于 debug
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = (object->d()).to_dict(); 
        }
        return output_dict;
    }

    py::object step(const py::object& actions) // 步进时会调用环境中所有对象的 step() 方法若干次，并获取返回值
    {
        py::dict result, info;

        for (int i = 0; i < integrate_steps; ++i)
        {
            for (auto& object : objects)
            {
                actions[py::str(object->name)]["dt"] = dt;   
                result[py::str(object->name)] = object->step(actions[py::str(object->name)]);
            }
        }

        return result;
    }
};
```