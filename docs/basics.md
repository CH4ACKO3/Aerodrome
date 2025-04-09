## 基础运行逻辑 ##

在 Aerodrome 中，一个典型的 C++/Python 联合编程环境的结构如图：
<figure markdown="span">
  ![Image title](./assets/minimal_example1.svg)
</figure>

通常来说，C++ 环境类直接管理 C++ 实体的初始化、运行，并在需要时读取其信息并输出；
Python 环境类则负责处理用户输入、C++ 环境的输出（例如，数据类型的必要转换）等；

当然，这种设计并不是强制的，用户完全可以根据需要自行设计，例如下面两种都是可行的：

<figure markdown="span">
  ![Image title](./assets/minimal_example2.svg)
  <figcaption>跳过 Python 环境类，直接从 C++ 环境类获取输出</figcaption>
</figure>

<figure markdown="span">
  ![Image title](./assets/minimal_example3.svg)
  <figcaption>在 Python 中直接实现所有功能</figcaption>
</figure>

# C++ 基础类 #
在 Aerodrome 中，有两个最基本的类型，可以在 `src/simulator/Core` 中找到：

- `#!cpp BaseEnv`：所有环境的基类，定义了所有环境共有的接口；
- `#!cpp Object3D`：所有三维对象的基类，定义了三维对象的通用属性，例如速度、位置、姿态等；

!!! info "关于 pybind11 代码实现"
    由于 pybind11 在编译方面的限制，目前 C++ 代码 pybind 绑定的实现方式是将函数/类的定义和实现都写在 `.h` 文件中；pybind 绑定则写在同名的 `.cpp` 文件中。

    如果有更好的实现方式，非常欢迎提出 issue 或 PR！

## BaseEnv ##
在 BaseEnv 中，定义了所有环境共有的接口，包括：

- `#!cpp reset()`：重置环境；
- `#!cpp step(action)`：执行一步动作，并返回信息（例如，观测、奖励、终止等）；

<details>
<summary>点击展开代码</summary>

```cpp title="BaseEnv.h"
class BaseEnv
{
public:
    BaseEnv() = default;
    virtual ~BaseEnv() = default;

    virtual py::object reset() = 0;
    virtual py::object step(const py::object& action) = 0;
};
```
</details>

## Object3D ##
Object3D 的内容则较多，包括：

- 位置、速度、姿态等基本属性；
- 将自身属性输出为字典的方法；
- 重置自身属性的方法；
- 计算自身动力学导数的方法；
- 进行一步仿真的方法；

为了准确方便地计算自身的动力学导数并进行仿真，还对运算符进行了重载。
在后面会看到，对于派生类，代码很大程度上是可以复用的，只需要重写少量代码就可以实现一个正确且规范的物理对象。

!!! warning "关于姿态表示"
    Aerodrome 中采用的姿态描述是欧拉角法，会有死锁问题；你也可以自己实现一个旋转矩阵或四元数等其它表示法。

<details>
<summary>点击展开代码</summary>

```C++ title="Object3D.h"
class Object3D {
public:
    std::string name;
    std::string integrator; // 积分器类型
    double dt;

    double init_m, m, d_m; // 初始质量；质量；质量变化率
    Eigen::Matrix3d J; // 惯性矩阵
    Eigen::Matrix3d J_inv; // 惯性矩阵的逆

    Eigen::Vector3d init_pos, pos; // 位置 (地面系)
    Eigen::Vector3d init_vel, vel; // 速度 (地面系)
    Eigen::Vector3d init_ang_vel, ang_vel; // 角速度 (弹体系)

    double init_V, V; // 速度
    double init_theta, theta;    // 俯仰角
    double init_phi, phi;      // 偏航角
    double init_gamma, gamma;    // 倾斜角
    double init_theta_v, theta_v;  // 速度倾角
    double init_phi_v, phi_v;    // 速度偏角
    double alpha;    // 攻角
    double beta;     // 侧滑角
    double gamma_v;  // 速度倾斜角

    typedef Eigen::Matrix<double, 10, 1> state_vec; // V, theta_v, phi_v, theta, phi, gamma, p, q, r, m
    typedef Eigen::Matrix<double, 6, 1> force_vec; // fx, fy, fz, mx, my, mz

    Object3D() {}

    Object3D(py::dict input_dict)
    {
        name = input_dict["name"].cast<std::string>();
        integrator = input_dict["integrator"].cast<std::string>();
        dt = input_dict["dt"].cast<double>();

        init_m = m = input_dict["m"].cast<double>();
        d_m = 0;
        std::array<double, 3> pos_ = input_dict["pos"].cast<std::array<double, 3>>();
        init_pos = pos = Eigen::Map<Eigen::Vector3d>(pos_.data());
        std::array<double, 3> vel_ = input_dict["vel"].cast<std::array<double, 3>>();
        init_vel = vel = Eigen::Map<Eigen::Vector3d>(vel_.data());
        std::array<double, 3> ang_vel_ = input_dict["ang_vel"].cast<std::array<double, 3>>();
        init_ang_vel = ang_vel = Eigen::Map<Eigen::Vector3d>(ang_vel_.data());
        std::array<double, 9> J_ = input_dict["J"].cast<std::array<double, 9>>();
        J = Eigen::Map<Eigen::Matrix3d>(J_.data());
        J_inv = J.inverse();
        init_V = V = sqrt(vel(0) * vel(0) + vel(1) * vel(1) + vel(2) * vel(2));
        init_theta = theta = input_dict["theta"].cast<double>();
        init_phi = phi = input_dict["phi"].cast<double>();
        init_gamma = gamma = input_dict["gamma"].cast<double>();
        init_theta_v = theta_v = input_dict["theta_v"].cast<double>();
        init_phi_v = phi_v = input_dict["phi_v"].cast<double>();

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));
        gamma_v = asin((cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v));
    }

    virtual void reset()
    {
        pos = init_pos;
        vel = init_vel;
        ang_vel = init_ang_vel;
        m = init_m;
        d_m = 0;

        V = init_V;
        theta = init_theta;
        phi = init_phi;
        gamma = init_gamma;
        theta_v = init_theta_v;
        phi_v = init_phi_v;

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));
        gamma_v = asin((cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v));
    }

    virtual py::dict to_dict()
    {
        py::dict output_dict;
        output_dict["pos"] = pos;
        output_dict["vel"] = vel;
        output_dict["ang_vel"] = ang_vel;
        output_dict["m"] = m;
        output_dict["dt"] = dt;
        output_dict["J"] = J;
        output_dict["J_inv"] = J_inv;
        output_dict["V"] = V;
        output_dict["theta"] = theta;
        output_dict["phi"] = phi;
        output_dict["gamma"] = gamma;
        output_dict["theta_v"] = theta_v;
        output_dict["phi_v"] = phi_v;
        output_dict["alpha"] = alpha;
        output_dict["beta"] = beta;
        output_dict["gamma_v"] = gamma_v;
        output_dict["name"] = name;
        return output_dict;
    }

    virtual py::object step(py::dict action)
    {
        double fx = 0, fy = 0, fz = 0; // 力 (弹体系)
        double mx = 0, my = 0, mz = 0; // 力矩 (弹体系)

        force_vec c_force = {fx, fy, fz, mx, my, mz}; // 力和力矩
        kinematics_step(c_force); // 更新状态

        return to_dict();
    }
    
    void kinematics_step(const force_vec& c_force)
    {
        state_vec c_state = {V, theta_v, phi_v,
                             theta, phi, gamma,
                             ang_vel(0), ang_vel(1), ang_vel(2), m};
        state_vec new_state;
        new_state.setZero();

        if (integrator == "euler")
        {
            // Update state using Euler method
            state_vec d_state = d(c_state, c_force);

            new_state = c_state + d_state * dt;
        }
        else if (integrator == "midpoint")
        {
            // Midpoint method
            state_vec k1 = d(c_state, c_force);
            state_vec k2 = d(c_state + k1 * (dt / 2), c_force);

            new_state = c_state + k2 * dt;
        }
        else if (integrator == "rk4")
        {
            // RK4 method
            state_vec k1 = d(c_state, c_force);
            state_vec k2 = d(c_state + k1 * (dt / 2), c_force);
            state_vec k3 = d(c_state + k2 * (dt / 2), c_force);
            state_vec k4 = d(c_state + k3 * dt, c_force);

            new_state = c_state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6);
        }
        
        V = new_state(0);
        theta_v = new_state(1);
        phi_v = new_state(2);
        theta = new_state(3);
        phi = new_state(4);
        gamma = new_state(5);
        ang_vel(0) = new_state(6);
        ang_vel(1) = new_state(7);
        ang_vel(2) = new_state(8);
        m = new_state(9);

        // [V, theta_v, phi_v,
        //  theta, phi, gamma,
        //  ang_vel(0), ang_vel(1), ang_vel(2), m] = new_state;

        vel(0) = V * cos(theta_v) * cos(phi_v);
        vel(1) = V * sin(theta_v);
        vel(2) = -V * cos(theta_v) * sin(phi_v);
        pos += vel * dt;

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));
        gamma_v = asin((cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v));
    }

    state_vec d(const state_vec& c_state, const force_vec& c_force) const
    {
        // c_x: current value of x
        // d_x: derivative of x

        auto c_V = c_state(0);
        auto c_theta_v = c_state(1);
        auto c_phi_v = c_state(2);
        auto c_theta = c_state(3);
        auto c_phi = c_state(4);
        auto c_gamma = c_state(5);
        auto c_p = c_state(6);
        auto c_q = c_state(7);
        auto c_r = c_state(8);
        auto c_m = c_state(9);

        double c_fx = c_force(0), c_fy = c_force(1), c_fz = c_force(2);
        double c_mx = c_force(3), c_my = c_force(4), c_mz = c_force(5);

        // Eigen::Vector 不支持结构化绑定
        // auto [c_V, c_theta_v, c_phi_v, c_theta, c_phi, c_gamma, c_p, c_q, c_r, c_m] = c_state;
        // auto [c_fx, c_fy, c_fz, c_mx, c_my, c_mz] = c_force;

        auto d_V = c_fx / c_m;
        auto d_theta_v = c_fy / (c_m * c_V);
        auto d_phi_v = - c_fz / (c_m * c_V * cos(c_theta_v));

        Eigen::Vector3d c_ang_vel(c_p, c_q, c_r);
        Eigen::Vector3d c_moment(c_mx, c_my, c_mz);
        Eigen::Vector3d d_ang_vel = J_inv * (c_moment - c_ang_vel.cross(J * c_ang_vel));

        auto d_theta = c_q * sin(c_gamma) + c_r * cos(c_gamma);
        auto d_phi = (c_q * cos(c_gamma) - c_r * sin(c_gamma)) / cos(c_theta);
        auto d_gamma = c_p - tan(c_theta) * (c_q * cos(c_gamma) - c_r * sin(c_gamma));

        state_vec d_state = {d_V, d_theta_v, d_phi_v,
                             d_theta, d_phi, d_gamma,
                             d_ang_vel(0), d_ang_vel(1), d_ang_vel(2), d_m};
        
        return d_state;
    }
};
```
</details>