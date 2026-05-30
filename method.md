# Method: 孔弹性波方程的指数缩放混合精度推导

本文档用于说明如何将 paper1.pdf 中基于二进制指数的缩放思想推广到 Zhang.pdf 中使用的三维低频 Biot 孔弹性波方程，并给出其在 `main.cu` 中传统单精度交错网格有限差分程序上的具体系数改写方式与缩放因子数值。

---

## 1. 目标与基本思想

Zhang.pdf 中原始 BEHP 方法采用线性缩放：

\[
\mathbf V_u=C_u\mathbf v_u,\qquad
\mathbf V_\omega=C_\omega\mathbf v_\omega,\qquad
P=C_p p,\qquad
\mathbf T=C_\tau\boldsymbol\tau .
\]

该方法可以将孔弹性波变量压入 FP16 表示范围，但需要引入较多人工调整系数，例如

\[
C_\xi,C_\zeta,C_\varsigma,C_l,C_\varpi,C_\varphi,C_\psi,C_\delta,C_\gamma,C_\sigma .
\]

这些系数在更换模型参数、孔隙度、饱和度、渗透率或时间步长后通常需要重新检查，使用上较麻烦。

paper1.pdf 的核心思想是：不直接使用任意线性缩放因子，而是使用

\[
2^e
\]

形式的指数缩放。由于 FP16 本身是二进制浮点格式，采用 \(2^e\) 形式的缩放更自然，也更便于自动化选择缩放因子。

本文将这一思想推广到孔弹性波方程，目标是将原来的多个线性调整因子简化为少数几个全局指数因子：

\[
e_m^B,\qquad e_s^B,\qquad e_r^B .
\]

其中：

- \(e_m^B\)：控制模量类参数与密度/惯性类参数的量级平衡；
- \(e_s^B\)：控制应力、压力和速度波场变量的整体幅值缩放；
- \(e_r^B\)：参考 paper1.pdf 的思想，用于自动平衡大模量参数和小密度倒数参数。

---

## 2. 原始孔弹性波方程

本文以如下低频 Biot 孔弹性波方程为基础：

\[
\left\{
\begin{aligned}
& \frac{\partial \mathbf{v}_u}{\partial t}
=\rho^{-1}\left(\nabla \cdot\boldsymbol{\tau}-\rho_f\frac{\partial \mathbf{v}_\omega}{\partial t}\right), \\
& C_1\mathbf{v}_\omega+
\left(C_2-\frac{\rho_f^2}{\rho}\right)
\frac{\partial \mathbf{v}_\omega}{\partial t}
=-\nabla p-\frac{\rho_f}{\rho}\nabla\cdot\boldsymbol{\tau}, \\
& \frac{\partial p}{\partial t}
=-(C\nabla\cdot \mathbf{v}_u+M\nabla\cdot \mathbf{v}_\omega), \\
& \frac{\partial\boldsymbol{\tau}}{\partial t}
=(H-2\mu)(\nabla\cdot \mathbf{v}_u)\mathbf{I}
+C(\nabla\cdot  \mathbf{v}_\omega)\mathbf{I}
+\mu\left(\nabla \mathbf{v}_u+
\left(\nabla \mathbf{v}_u\right)^{\mathrm T}\right).
\end{aligned}
\right.
\]

其中：

\[
\left\{
\begin{aligned}
& C_1=\frac{\eta}{\kappa_0}, \\
& C_2=\alpha_\infty(1+2m^{-1})\frac{\rho_f}{\phi}, \\
& H=K_\mu+\frac{4}{3}\mu_s, \\
& C=\alpha M, \\
& M=\frac{K_fK_s}{\phi K_s+(\alpha-\phi)K_f}.
\end{aligned}
\right.
\]

为了简化推导，定义

\[
Q=C_2-\frac{\rho_f^2}{\rho} .
\]

则第二个方程可以写为

\[
Q\frac{\partial \mathbf v_\omega}{\partial t}+C_1\mathbf v_\omega
=-\nabla p-\rho_f\rho^{-1}\nabla\cdot\boldsymbol\tau .
\]

---

## 3. 指数缩放变量定义

将速度类变量和应力/压力类变量分别采用不同的指数缩放。定义

\[
\tilde{\mathbf v}_u=2^{e_s^B-e_m^B}\mathbf v_u,
\]

\[
\tilde{\mathbf v}_\omega=2^{e_s^B-e_m^B}\mathbf v_\omega,
\]

\[
\tilde p=2^{e_s^B}p,
\]

\[
\tilde{\boldsymbol\tau}=2^{e_s^B}\boldsymbol\tau .
\]

反变换为

\[
\mathbf v_u=2^{e_m^B-e_s^B}\tilde{\mathbf v}_u,
\]

\[
\mathbf v_\omega=2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega,
\]

\[
p=2^{-e_s^B}\tilde p,
\]

\[
\boldsymbol\tau=2^{-e_s^B}\tilde{\boldsymbol\tau} .
\]

这里令 \(\mathbf v_u\) 和 \(\mathbf v_\omega\) 使用相同的速度缩放，是因为二者在方程中均为速度型变量。令 \(p\) 和 \(\boldsymbol\tau\) 使用相同的缩放，是因为二者都是压力/应力型变量。

---

## 4. 第一方程的缩放推导：固相速度方程

原始方程为

\[
\frac{\partial \mathbf{v}_u}{\partial t}
=\rho^{-1}\nabla\cdot\boldsymbol\tau
-\rho_f\rho^{-1}\frac{\partial \mathbf v_\omega}{\partial t}.
\]

代入

\[
\mathbf v_u=2^{e_m^B-e_s^B}\tilde{\mathbf v}_u,
\qquad
\mathbf v_\omega=2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega,
\qquad
\boldsymbol\tau=2^{-e_s^B}\tilde{\boldsymbol\tau},
\]

得到

\[
2^{e_m^B-e_s^B}\frac{\partial \tilde{\mathbf v}_u}{\partial t}
=ho^{-1}2^{-e_s^B}\nabla\cdot\tilde{\boldsymbol\tau}
-\rho_f\rho^{-1}2^{e_m^B-e_s^B}
\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}.
\]

两边同乘 \(2^{e_s^B-e_m^B}\)，得到

\[
\frac{\partial \tilde{\mathbf v}_u}{\partial t}
=2^{-e_m^B}\rho^{-1}\nabla\cdot\tilde{\boldsymbol\tau}
-\rho_f\rho^{-1}
\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}.
\]

乘以时间步长 \(\Delta t\)，得到适合时间推进的形式：

\[
\Delta t\frac{\partial \tilde{\mathbf v}_u}{\partial t}
=\tilde\rho^{-1}\nabla\cdot\tilde{\boldsymbol\tau}
-\rho_f\rho^{-1}
\Delta t\frac{\partial \tilde{\mathbf v}_\omega}{\partial t},
\]

其中

\[
\boxed{
\tilde\rho^{-1}=2^{-e_m^B}\Delta t\rho^{-1}
}
\]

是缩放后的密度倒数系数。

---

## 5. 第二方程的缩放推导：相对流体速度方程

原始方程为

\[
Q\frac{\partial \mathbf v_\omega}{\partial t}+C_1\mathbf v_\omega
=-\nabla p-\rho_f\rho^{-1}\nabla\cdot\boldsymbol\tau .
\]

代入缩放变量：

\[
\mathbf v_\omega=2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega,
\qquad
p=2^{-e_s^B}\tilde p,
\qquad
\boldsymbol\tau=2^{-e_s^B}\tilde{\boldsymbol\tau}.
\]

得到

\[
Q2^{e_m^B-e_s^B}\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}
+C_1 2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega
=-2^{-e_s^B}\nabla\tilde p
-\rho_f\rho^{-1}2^{-e_s^B}\nabla\cdot\tilde{\boldsymbol\tau}.
\]

两边同乘 \(2^{e_s^B-e_m^B}\)，得到

\[
Q\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}
+C_1\tilde{\mathbf v}_\omega
=-2^{-e_m^B}\nabla\tilde p
-2^{-e_m^B}\rho_f\rho^{-1}\nabla\cdot\tilde{\boldsymbol\tau}.
\]

除以 \(Q\)：

\[
\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}
=-\frac{C_1}{Q}\tilde{\mathbf v}_\omega
-2^{-e_m^B}\frac{1}{Q}\nabla\tilde p
-2^{-e_m^B}\frac{\rho_f}{\rho Q}\nabla\cdot\tilde{\boldsymbol\tau}.
\]

乘以 \(\Delta t\)，得到

\[
\Delta t\frac{\partial \tilde{\mathbf v}_\omega}{\partial t}
=-\tilde C_1^Q\tilde{\mathbf v}_\omega
-\tilde Q^{-1}\nabla\tilde p
-\tilde R_Q\nabla\cdot\tilde{\boldsymbol\tau},
\]

其中

\[
\boxed{
\tilde C_1^Q=\Delta t\frac{C_1}{Q}
}
\]

\[
\boxed{
\tilde Q^{-1}=2^{-e_m^B}\Delta t\frac{1}{Q}
}
\]

\[
\boxed{
\tilde R_Q=2^{-e_m^B}\Delta t\frac{\rho_f}{\rho Q}
}
\]

需要注意：\(\tilde C_1^Q\) 是阻尼型无量纲系数，它本身不含应力-速度量纲转换，因此不受 \(2^{e_m^B}\) 或 \(2^{-e_m^B}\) 控制。如果 \(\Delta t C_1/Q\) 过大，需要单独处理，例如采用 FP32 计算、半精度存储，或再引入阻尼项局部缩放因子。

---

## 6. 第三方程的缩放推导：孔隙压力方程

原始方程为

\[
\frac{\partial p}{\partial t}
=-(C\nabla\cdot\mathbf v_u+M\nabla\cdot\mathbf v_\omega).
\]

代入

\[
p=2^{-e_s^B}\tilde p,
\qquad
\mathbf v_u=2^{e_m^B-e_s^B}\tilde{\mathbf v}_u,
\qquad
\mathbf v_\omega=2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega .
\]

得到

\[
2^{-e_s^B}\frac{\partial \tilde p}{\partial t}
=-C2^{e_m^B-e_s^B}\nabla\cdot\tilde{\mathbf v}_u
-M2^{e_m^B-e_s^B}\nabla\cdot\tilde{\mathbf v}_\omega .
\]

两边同乘 \(2^{e_s^B}\)，得到

\[
\frac{\partial \tilde p}{\partial t}
=-2^{e_m^B}C\nabla\cdot\tilde{\mathbf v}_u
-2^{e_m^B}M\nabla\cdot\tilde{\mathbf v}_\omega .
\]

乘以 \(\Delta t\)：

\[
\Delta t\frac{\partial \tilde p}{\partial t}
=-(\tilde C\nabla\cdot\tilde{\mathbf v}_u
+\tilde M\nabla\cdot\tilde{\mathbf v}_\omega),
\]

其中

\[
\boxed{
\tilde C=2^{e_m^B}\Delta t C
}
\]

\[
\boxed{
\tilde M=2^{e_m^B}\Delta t M
}
\]

---

## 7. 第四方程的缩放推导：应力方程

原始应力方程为

\[
\frac{\partial\boldsymbol{\tau}}{\partial t}
=(H-2\mu)(\nabla\cdot\mathbf v_u)\mathbf I
+C(\nabla\cdot\mathbf v_\omega)\mathbf I
+\mu\left[\nabla\mathbf v_u+(\nabla\mathbf v_u)^T\right].
\]

代入

\[
\boldsymbol\tau=2^{-e_s^B}\tilde{\boldsymbol\tau},
\qquad
\mathbf v_u=2^{e_m^B-e_s^B}\tilde{\mathbf v}_u,
\qquad
\mathbf v_\omega=2^{e_m^B-e_s^B}\tilde{\mathbf v}_\omega .
\]

得到

\[
2^{-e_s^B}\frac{\partial\tilde{\boldsymbol\tau}}{\partial t}
=(H-2\mu)2^{e_m^B-e_s^B}(\nabla\cdot\tilde{\mathbf v}_u)\mathbf I
+C2^{e_m^B-e_s^B}(\nabla\cdot\tilde{\mathbf v}_\omega)\mathbf I
+\mu2^{e_m^B-e_s^B}
\left[\nabla\tilde{\mathbf v}_u+(\nabla\tilde{\mathbf v}_u)^T\right].
\]

两边同乘 \(2^{e_s^B}\)：

\[
\frac{\partial\tilde{\boldsymbol\tau}}{\partial t}
=2^{e_m^B}(H-2\mu)(\nabla\cdot\tilde{\mathbf v}_u)\mathbf I
+2^{e_m^B}C(\nabla\cdot\tilde{\mathbf v}_\omega)\mathbf I
+2^{e_m^B}\mu
\left[\nabla\tilde{\mathbf v}_u+(\nabla\tilde{\mathbf v}_u)^T\right].
\]

乘以 \(\Delta t\)：

\[
\Delta t\frac{\partial\tilde{\boldsymbol\tau}}{\partial t}
=\tilde H_\mu(\nabla\cdot\tilde{\mathbf v}_u)\mathbf I
+\tilde C(\nabla\cdot\tilde{\mathbf v}_\omega)\mathbf I
+\tilde\mu
\left[\nabla\tilde{\mathbf v}_u+(\nabla\tilde{\mathbf v}_u)^T\right],
\]

其中

\[
\boxed{
\tilde H_\mu=2^{e_m^B}\Delta t(H-2\mu)
}
\]

\[
\boxed{
\tilde C=2^{e_m^B}\Delta t C
}
\]

\[
\boxed{
\tilde\mu=2^{e_m^B}\Delta t\mu
}
\]

---

## 8. 张量分量展开形式

在三维交错网格有限差分程序中，通常将应力分量写成如下形式。

### 8.1 压力更新

\[
\tilde p^{n+1}
=\tilde p^n
-\tilde C\left(
\frac{\partial \tilde v_{ux}}{\partial x}
+\frac{\partial \tilde v_{uy}}{\partial y}
+\frac{\partial \tilde v_{uz}}{\partial z}
\right)
-\tilde M\left(
\frac{\partial \tilde v_{\omega x}}{\partial x}
+\frac{\partial \tilde v_{\omega y}}{\partial y}
+\frac{\partial \tilde v_{\omega z}}{\partial z}
\right).
\]

### 8.2 正应力更新

在 `main.cu` 中，正应力更新采用的是等价的分量形式：

\[
\tau_{xx}^{n+1}
=\tau_{xx}^{n}
+\Delta t\left[H\frac{\partial v_{ux}}{\partial x}
+(H-2\mu)\left(
\frac{\partial v_{uy}}{\partial y}
+\frac{\partial v_{uz}}{\partial z}
\right)
+C\nabla\cdot\mathbf v_\omega
\right].
\]

因此缩放后写为

\[
\tilde\tau_{xx}^{n+1}
=\tilde\tau_{xx}^{n}
+\tilde H\frac{\partial \tilde v_{ux}}{\partial x}
+\tilde H_{2\mu}\left(
\frac{\partial \tilde v_{uy}}{\partial y}
+\frac{\partial \tilde v_{uz}}{\partial z}
\right)
+\tilde C\nabla\cdot\tilde{\mathbf v}_\omega,
\]

其中

\[
\boxed{
\tilde H=2^{e_m^B}\Delta t H
}
\]

\[
\boxed{
\tilde H_{2\mu}=2^{e_m^B}\Delta t(H-2\mu)
}
\]

同理，

\[
\tilde\tau_{yy}^{n+1}
=\tilde\tau_{yy}^{n}
+\tilde H\frac{\partial \tilde v_{uy}}{\partial y}
+\tilde H_{2\mu}\left(
\frac{\partial \tilde v_{ux}}{\partial x}
+\frac{\partial \tilde v_{uz}}{\partial z}
\right)
+\tilde C\nabla\cdot\tilde{\mathbf v}_\omega,
\]

\[
\tilde\tau_{zz}^{n+1}
=\tilde\tau_{zz}^{n}
+\tilde H\frac{\partial \tilde v_{uz}}{\partial z}
+\tilde H_{2\mu}\left(
\frac{\partial \tilde v_{ux}}{\partial x}
+\frac{\partial \tilde v_{uy}}{\partial y}
\right)
+\tilde C\nabla\cdot\tilde{\mathbf v}_\omega.
\]

### 8.3 剪应力更新

\[
\tilde\tau_{xy}^{n+1}
=\tilde\tau_{xy}^{n}
+\tilde\mu_{xy}\left(
\frac{\partial \tilde v_{ux}}{\partial y}
+\frac{\partial \tilde v_{uy}}{\partial x}
\right),
\]

\[
\tilde\tau_{xz}^{n+1}
=\tilde\tau_{xz}^{n}
+\tilde\mu_{xz}\left(
\frac{\partial \tilde v_{ux}}{\partial z}
+\frac{\partial \tilde v_{uz}}{\partial x}
\right),
\]

\[
\tilde\tau_{yz}^{n+1}
=\tilde\tau_{yz}^{n}
+\tilde\mu_{yz}\left(
\frac{\partial \tilde v_{uy}}{\partial z}
+\frac{\partial \tilde v_{uz}}{\partial y}
\right),
\]

其中

\[
\boxed{
\tilde\mu_{xy}=2^{e_m^B}\Delta t\mu_{xy},\qquad
\tilde\mu_{xz}=2^{e_m^B}\Delta t\mu_{xz},\qquad
\tilde\mu_{yz}=2^{e_m^B}\Delta t\mu_{yz}.
}
\]

---

## 9. 指数缩放因子的自动选择

为了让模量类项和密度/惯性类项在 FP16 中具有相近量级，定义模量类最大值

\[
K_{\max}
=\max\left(
|H|, |H-2\mu|, |C|, |M|, |\mu|
\right).
\]

其中 \(|H|\) 被加入，是因为 `main.cu` 中正应力更新直接使用 `HH_ext` 作为 \(H\) 的系数。

定义惯性类最大值

\[
R_{\max}
=\max\left(
|\rho^{-1}|,
|Q^{-1}|,
\left|\frac{\rho_f}{\rho Q}\right|
\right).
\]

然后参考 paper1.pdf 的指数缩放思想，引入

\[
\boxed{
 e_r^B=\left(\Delta t^2K_{\max}R_{\max}\right)^{-1/2}
}
\]

并定义

\[
\boxed{
 e_m^B=-\log_2\left(e_r^B\Delta tK_{\max}\right)
}
\]

实际代码中建议使用整数指数

\[
\boxed{
 e_{m,\mathrm{int}}^B=\mathrm{round}(e_m^B)
}
\]

或者为了避免最小系数过小，可使用

\[
 e_{m,\mathrm{int}}^B=\lceil e_m^B\rceil
\quad \text{或} \quad
 e_{m,\mathrm{int}}^B=\lfloor e_m^B\rfloor
\]

并检查所有缩放后的系数是否落在 FP16 安全范围内。

如果应力源或压力源具有明确物理量纲，可以进一步定义

\[
\boxed{
 e_s^B=-\log_2\left(e_r^B\Delta tS_{\max}\right)
}
\]

其中 \(S_{\max}\) 为源项最大幅值。但是在 `main.cu` 当前程序中，震源 `I_sou` 是直接加到 `txx, tyy, tzz` 上的无量纲 Ricker 波形，幅值约为 1，因此本文建议先取

\[
\boxed{e_s^B=0}
\]

这样可以最大限度减少对原始源函数的改动。如果后续将震源改为具有 Pa/s 或 Pa 量纲的物理源项，再重新启用 \(e_s^B\) 的自动计算。

---

## 10. 与 `main.cu` 的变量对应关系

`main.cu` 中主要物理量与理论符号对应如下：

| 理论符号 | `main.cu` 变量 | 含义 |
|---|---|---|
| \(\mathbf v_u\) | `vux, vuy, vuz` | 固相速度 |
| \(\mathbf v_\omega\) | `vwx, vwy, vwz` | 相对流体速度 |
| \(p\) | `ss` | 孔隙压力变量 |
| \(\tau_{xx},\tau_{yy},\tau_{zz}\) | `txx, tyy, tzz` | 正应力 |
| \(\tau_{xy},\tau_{xz},\tau_{yz}\) | `txy, txz, tyz` | 剪应力 |
| \(\rho^{-1}\) | `rho_tempx, rho_tempy, rho_tempz` | 交错网格上的密度倒数 |
| \(\rho_f\) | `rhof_ext` | 流体密度 |
| \(C_1\) | `C1x, C1y, C1z` | 黏滞阻尼系数 |
| \(C_2\) | `C2x, C2y, C2z` | 惯性耦合系数 |
| \(H\) | `HH_ext` | 正应力主方向系数 |
| \(H-2\mu\) | `H2u_ext` | 正应力交叉方向系数 |
| \(C\) | `C_ext` | 固-流耦合模量 |
| \(M\) | `M_ext` | Biot 模量 |
| \(\mu\) | `muxy, muxz, muyz` | 交错网格剪切模量 |

---

## 11. `main.cu` 中速度更新方程的指数缩放形式

### 11.1 原始 `vwx` 更新

`main.cu` 中相对流体速度 `vwx` 的更新形式为

```c
CC1x = 0.5 * C1x[offset] * DT + C2x[offset]
       - rhof_ext[offset] * rhof_ext[offset] * rho_tempx[offset];

CC2x = C2x[offset]
       - rhof_ext[offset] * rhof_ext[offset] * rho_tempx[offset]
       - 0.5 * C1x[offset] * DT;

vwx[offset] = (vwx[offset] * CC2x
              - DT * (rhof_ext[offset] * rho_tempx[offset]
              * (x1 + x2 + x3) + s1)) / CC1x;
```

其中

\[
Q_x=C_{2x}-\rho_f^2\rho_x^{-1}.
\]

原始格式等价于

\[
\left(Q_x+\frac{1}{2}C_{1x}\Delta t\right)v_{\omega x}^{n+1}
=
\left(Q_x-\frac{1}{2}C_{1x}\Delta t\right)v_{\omega x}^{n}
-
\Delta t\left(\rho_f\rho_x^{-1}\nabla\cdot\boldsymbol\tau_x+rac{\partial p}{\partial x}\right).
\]

缩放后，应写为

\[
\left(Q_x+\frac{1}{2}C_{1x}\Delta t\right)\tilde v_{\omega x}^{n+1}
=
\left(Q_x-\frac{1}{2}C_{1x}\Delta t\right)\tilde v_{\omega x}^{n}
-
2^{-e_m^B}\Delta t
\left(\rho_f\rho_x^{-1}\nabla\cdot\tilde{\boldsymbol\tau}_x+rac{\partial \tilde p}{\partial x}\right).
\]

为了避免在 FP16 中直接存储大数 \(Q_x\)，推荐预先计算三个无量纲系数：

\[
\boxed{
A_{\omega x}=\frac{Q_x-\frac{1}{2}C_{1x}\Delta t}
{Q_x+\frac{1}{2}C_{1x}\Delta t}
}
\]

\[
\boxed{
B_{px}=\frac{2^{-e_m^B}\Delta t}
{Q_x+\frac{1}{2}C_{1x}\Delta t}
}
\]

\[
\boxed{
B_{\tau x}=\frac{2^{-e_m^B}\Delta t\rho_f\rho_x^{-1}}
{Q_x+\frac{1}{2}C_{1x}\Delta t}
}
\]

则

\[
\boxed{
\tilde v_{\omega x}^{n+1}
=A_{\omega x}\tilde v_{\omega x}^{n}
-B_{\tau x}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_x
-B_{px}\frac{\partial \tilde p}{\partial x}
}
\]

同理，\(y,z\) 方向分别为

\[
\tilde v_{\omega y}^{n+1}
=A_{\omega y}\tilde v_{\omega y}^{n}
-B_{\tau y}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_y
-B_{py}\frac{\partial \tilde p}{\partial y},
\]

\[
\tilde v_{\omega z}^{n+1}
=A_{\omega z}\tilde v_{\omega z}^{n}
-B_{\tau z}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_z
-B_{pz}\frac{\partial \tilde p}{\partial z}.
\]

### 11.2 原始 `vux` 更新

`main.cu` 中固相速度 `vux` 的更新为

```c
vux[offset] = vux[offset]
            + DT * rho_tempx[offset] * (x1 + x2 + x3)
            - rhof_ext[offset] * rho_tempx[offset]
              * (vwx[offset] - vwx2[offset]);
```

缩放后为

\[
\boxed{
\tilde v_{ux}^{n+1}
=\tilde v_{ux}^{n}
+\tilde\rho_x^{-1}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_x
-\rho_f\rho_x^{-1}
\left(\tilde v_{\omega x}^{n+1}-\tilde v_{\omega x}^{n}\right)
}
\]

其中

\[
\tilde\rho_x^{-1}=2^{-e_m^B}\Delta t\rho_x^{-1}.
\]

同理，

\[
\tilde v_{uy}^{n+1}
=\tilde v_{uy}^{n}
+\tilde\rho_y^{-1}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_y
-\rho_f\rho_y^{-1}
\left(\tilde v_{\omega y}^{n+1}-\tilde v_{\omega y}^{n}\right),
\]

\[
\tilde v_{uz}^{n+1}
=\tilde v_{uz}^{n}
+\tilde\rho_z^{-1}\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_z
-\rho_f\rho_z^{-1}
\left(\tilde v_{\omega z}^{n+1}-\tilde v_{\omega z}^{n}\right).
\]

---

## 12. `main.cu` 中应力和压力更新的指数缩放形式

原始代码为

```c
ss[offset] = ss[offset]
           - DT * (C_ext[offset] * (uxx + uyy + uzz)
           + M_ext[offset] * (wx + wy + wz));

txx[offset] = txx[offset]
            + DT * (H2u_ext[offset] * (uyy + uzz)
            + HH_ext[offset] * uxx
            + C_ext[offset] * (wx + wy + wz));

tyy[offset] = tyy[offset]
            + DT * (H2u_ext[offset] * (uxx + uzz)
            + HH_ext[offset] * uyy
            + C_ext[offset] * (wx + wy + wz));

tzz[offset] = tzz[offset]
            + DT * (H2u_ext[offset] * (uxx + uyy)
            + HH_ext[offset] * uzz
            + C_ext[offset] * (wx + wy + wz));

txy[offset] = txy[offset] + muxy[offset] * DT * (uxy + uyx);
tyz[offset] = tyz[offset] + muyz[offset] * DT * (uyz + uzy);
txz[offset] = txz[offset] + muxz[offset] * DT * (uxz + uzx);
```

缩放后将所有模量型系数预先改写为

\[
\boxed{
\tilde C=2^{e_m^B}\Delta tC
}
\]

\[
\boxed{
\tilde M=2^{e_m^B}\Delta tM
}
\]

\[
\boxed{
\tilde H=2^{e_m^B}\Delta tH
}
\]

\[
\boxed{
\tilde H_{2\mu}=2^{e_m^B}\Delta t(H-2\mu)
}
\]

\[
\boxed{
\tilde\mu_{xy}=2^{e_m^B}\Delta t\mu_{xy},\qquad
\tilde\mu_{xz}=2^{e_m^B}\Delta t\mu_{xz},\qquad
\tilde\mu_{yz}=2^{e_m^B}\Delta t\mu_{yz}.
}
\]

更新式改为

```c
ss_tilde[offset] = ss_tilde[offset]
                 - C_tilde[offset] * (uxx_tilde + uyy_tilde + uzz_tilde)
                 - M_tilde[offset] * (wx_tilde + wy_tilde + wz_tilde);

txx_tilde[offset] = txx_tilde[offset]
                  + H2u_tilde[offset] * (uyy_tilde + uzz_tilde)
                  + HH_tilde[offset]  * uxx_tilde
                  + C_tilde[offset]   * (wx_tilde + wy_tilde + wz_tilde);

txy_tilde[offset] = txy_tilde[offset]
                  + muxy_tilde[offset] * (uxy_tilde + uyx_tilde);
```

注意：此时系数中已经包含 \(\Delta t\)，因此更新式中不再额外乘以 `DT`。

---

## 13. PML 辅助变量的缩放说明

`main.cu` 中 PML 辅助变量如 `pmlxSxx, pmlySxy, pmlzSxz` 存储的是应力空间导数的历史记忆项，例如

```c
pmlxSxx[offset] = pmlxSxx[offset] * e_dxi2[offset]
                + (-DT * dxi2[offset] * 0.5)
                * (e_dxi2[offset] * SXxx[offset] + x1);
```

在指数缩放后，`x1` 已经由真实应力导数变为缩放应力导数：

\[
x_1=\frac{\partial\tilde\tau_{xx}}{\partial x}.
\]

因此 PML 中对应的历史变量也应当存储缩放后的导数记忆项。其形式不需要额外乘以 \(2^{e_m^B}\) 或 \(2^{e_s^B}\)，只需要保证参与更新的波场变量均为缩放变量即可。

同理，速度导数类 PML 变量如 `pmlxVux, pmlyVuy, pmlzVuz` 也直接使用缩放速度导数。

---

## 14. 根据 `main.cu` 计算具体缩放因子

### 14.1 `main.cu` 中读取到的关键数值

从 `main.cu` 中提取到：

| 参数 | 数值 |
|---|---:|
| 网格规模 | \(128\times128\times128\) |
| PML 厚度 | \(NP=32\) |
| 空间步长 | \(H=0.01\ \mathrm{m}\) |
| 时间步长 | \(\Delta t=0.9\times10^{-6}\ \mathrm{s}\) |
| 时间步数 | \(NT=3500\) |
| 主频 | \(F_0=3000\ \mathrm{Hz}\) |
| 固体颗粒纵波速度 | \(v_{ps}=6500\ \mathrm{m/s}\) |
| 固体颗粒横波速度 | \(v_{ss}=4000\ \mathrm{m/s}\) |
| 固体颗粒密度 | \(\rho_s=3200\ \mathrm{kg/m^3}\) |
| 地层纵波速度 | \(v_p=4000\ \mathrm{m/s}\) |
| 地层横波速度 | \(v_s=2700\ \mathrm{m/s}\) |
| 地层固体密度 | \(\rho_s=2650\ \mathrm{kg/m^3}\) |
| 流体密度 | \(\rho_f=1000\ \mathrm{kg/m^3}\) |
| 流体速度 | \(v_f=1500\ \mathrm{m/s}\) |
| 孔隙度 | \(\phi=0.1\) |
| 渗透率 | \(\kappa_0=2\times10^{-12}\ \mathrm{m^2}\) |
| 黏度 | \(\eta=10^{-3}\ \mathrm{Pa\cdot s}\) |
| 曲折度 | \(\tau=3\) |
| `porousm` | 8 |

### 14.2 地层参数计算

对于地层部分，`main.cu` 中有

\[
\rho=(1-\phi)\rho_s+\phi\rho_f.
\]

代入数值得到

\[
\rho=(1-0.1)\times2650+0.1\times1000=2485\ \mathrm{kg/m^3}.
\]

剪切模量为

\[
\mu=(1-\phi)\rho_s v_s^2.
\]

即

\[
\mu=0.9\times2650\times2700^2
=1.738665\times10^{10}\ \mathrm{Pa}.
\]

流体体积模量为

\[
K_f=\rho_f v_f^2=1000\times1500^2=2.25\times10^9\ \mathrm{Pa}.
\]

干岩石体积模量为

\[
K_b=\rho_s(1-\phi)\left(v_p^2-\frac{4}{3}v_s^2\right)
=1.49778\times10^{10}\ \mathrm{Pa}.
\]

固体颗粒体积模量为

\[
K_s=\rho_s\left(v_p^2-\frac{4}{3}v_s^2\right)
=1.66420\times10^{10}\ \mathrm{Pa}.
\]

Biot 系数为

\[
\alpha=1-\frac{K_b}{K_s}=0.1.
\]

Biot 模量为

\[
M=\frac{K_fK_s}{\phi K_s+(\alpha-\phi)K_f}.
\]

由于 \(\alpha=\phi=0.1\)，所以

\[
M=\frac{K_fK_s}{0.1K_s}=10K_f=2.25\times10^{10}\ \mathrm{Pa}.
\]

固-流耦合模量为

\[
C=\alpha M=0.1\times2.25\times10^{10}=2.25\times10^9\ \mathrm{Pa}.
\]

`main.cu` 中的 `HH` 为

\[
H=\alpha^2M+K_b+\frac{4}{3}\mu
=3.8385\times10^{10}\ \mathrm{Pa}.
\]

`H2u` 为

\[
H-2\mu=3.6117\times10^9\ \mathrm{Pa}.
\]

黏滞系数为

\[
C_1=\frac{\eta}{\kappa_0}
=\frac{10^{-3}}{2\times10^{-12}}
=5.0\times10^8.
\]

惯性耦合项为

\[
C_2=(1+2/8)\times3\times\frac{1000}{0.1}
=3.75\times10^4.
\]

\[
Q=C_2-\frac{\rho_f^2}{\rho}
=3.75\times10^4-\frac{1000^2}{2485}
=3.7097585513\times10^4.
\]

### 14.3 最大模量项

由于代码中正应力更新直接使用 `HH_ext`，因此取

\[
K_{\max}=\max(|H|,|H-2\mu|,|C|,|M|,|\mu|).
\]

代入数值：

\[
K_{\max}
=\max(
3.8385\times10^{10},
3.6117\times10^9,
2.25\times10^9,
2.25\times10^{10},
1.738665\times10^{10})
=3.8385\times10^{10}.
\]

### 14.4 最大惯性项

\[
R_{\max}=\max\left(
\rho^{-1}, Q^{-1}, \frac{\rho_f}{\rho Q}
\right).
\]

其中

\[
\rho^{-1}=\frac{1}{2485}=4.0241448692\times10^{-4},
\]

\[
Q^{-1}=\frac{1}{3.7097585513\times10^4}=2.6955921381\times10^{-5},
\]

\[
\frac{\rho_f}{\rho Q}=1.0853467849\times10^{-5}.
\]

因此

\[
R_{\max}=4.0241448692\times10^{-4}.
\]

### 14.5 计算指数缩放因子

\[
e_r^B=\left(\Delta t^2K_{\max}R_{\max}\right)^{-1/2}.
\]

代入

\[
\Delta t=0.9\times10^{-6},
\quad
K_{\max}=3.8385\times10^{10},
\quad
R_{\max}=4.0241448692\times10^{-4},
\]

得到

\[
\boxed{
e_r^B=2.8270918241\times10^2
}
\]

即

\[
e_r^B\approx 282.709.
\]

然后

\[
e_m^B=-\log_2(e_r^B\Delta tK_{\max}).
\]

计算得到

\[
\boxed{
e_m^B=-23.2194268667
}
\]

实际代码中采用整数指数。推荐取

\[
\boxed{
e_{m,\mathrm{int}}^B=-23
}
\]

此时

\[
\boxed{
2^{e_m^B}=2^{-23}=1.1920928955\times10^{-7}
}
\]

\[
\boxed{
2^{-e_m^B}=2^{23}=8.388608\times10^6
}
\]

由于当前 `main.cu` 中震源幅值约为 1，本文建议先取

\[
\boxed{
e_s^B=0
}
\]

因此

\[
\tilde{\mathbf v}_u=2^{23}\mathbf v_u,
\qquad
\tilde{\mathbf v}_\omega=2^{23}\mathbf v_\omega,
\qquad
\tilde p=p,
\qquad
\tilde{\boldsymbol\tau}=\boldsymbol\tau.
\]

反变换为

\[
\mathbf v_u=2^{-23}\tilde{\mathbf v}_u,
\qquad
\mathbf v_\omega=2^{-23}\tilde{\mathbf v}_\omega,
\qquad
p=\tilde p,
\qquad
\boldsymbol\tau=\tilde{\boldsymbol\tau}.
\]

---

## 15. 当前 `main.cu` 参数下的缩放后系数数值

采用

\[
e_m^B=-23,
\qquad
2^{e_m^B}=1.1920928955\times10^{-7},
\qquad
2^{-e_m^B}=8.388608\times10^6,
\qquad
\Delta t=0.9\times10^{-6}.
\]

### 15.1 密度和惯性类系数

\[
\tilde\rho^{-1}=2^{-e_m^B}\Delta t\rho^{-1}
=3.0381276459\times10^{-3}.
\]

\[
\tilde Q^{-1}=2^{-e_m^B}\Delta tQ^{-1}
=2.0351047368\times10^{-4}.
\]

\[
\tilde R_Q=2^{-e_m^B}\Delta t\frac{\rho_f}{\rho Q}
=8.1895562847\times10^{-5}.
\]

\[
\tilde C_1^Q=\Delta t\frac{C_1}{Q}
=1.2130169492\times10^{-2}.
\]

这些值均处于 FP16 正常表示范围附近或内部，且不会接近 FP16 上限。

### 15.2 模量类系数

\[
\tilde H=2^{e_m^B}\Delta tH
=4.1182637215\times10^{-3}.
\]

\[
\tilde H_{2\mu}=2^{e_m^B}\Delta t(H-2\mu)
=3.8749337196\times10^{-4}.
\]

\[
\tilde C=2^{e_m^B}\Delta tC
=2.4139881134\times10^{-4}.
\]

\[
\tilde M=2^{e_m^B}\Delta tM
=2.4139881134\times10^{-3}.
\]

\[
\tilde\mu=2^{e_m^B}\Delta t\mu
=1.8653851748\times10^{-3}.
\]

### 15.3 相对流体速度隐式阻尼更新系数

对地层区域，

\[
Q=3.7097585513\times10^4,
\qquad
\frac{1}{2}C_1\Delta t=225.
\]

因此

\[
Q+\frac{1}{2}C_1\Delta t=3.7322585513\times10^4,
\]

\[
Q-\frac{1}{2}C_1\Delta t=3.6872585513\times10^4.
\]

得到

\[
A_\omega
=\frac{Q-\frac{1}{2}C_1\Delta t}
{Q+\frac{1}{2}C_1\Delta t}
=0.9879429575.
\]

\[
B_p
=\frac{2^{-e_m^B}\Delta t}
{Q+\frac{1}{2}C_1\Delta t}
=2.0228360646\times10^{-4}.
\]

\[
B_\tau
=\frac{2^{-e_m^B}\Delta t\rho_f\rho^{-1}}
{Q+\frac{1}{2}C_1\Delta t}
=8.1401853706\times10^{-5}.
\]

固相速度更新中的耦合系数为

\[
\rho_f\rho^{-1}=\frac{1000}{2485}=0.4024144869.
\]

因此速度更新可以写成

\[
\tilde v_{\omega x}^{n+1}
=0.9879429575\tilde v_{\omega x}^{n}
-8.1401853706\times10^{-5}
\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_x
-2.0228360646\times10^{-4}
\frac{\partial \tilde p}{\partial x},
\]

\[
\tilde v_{ux}^{n+1}
=\tilde v_{ux}^{n}
+3.0381276459\times10^{-3}
\left(\nabla\cdot\tilde{\boldsymbol\tau}\right)_x
-0.4024144869
\left(
\tilde v_{\omega x}^{n+1}-\tilde v_{\omega x}^{n}
\right).
\]

\(y,z\) 方向同理，只是使用对应交错网格位置上的 \(\rho_y^{-1},\rho_z^{-1},C_{1y},C_{1z},C_{2y},C_{2z}\)。

---

## 16. 纯流体井孔区域的特殊处理

`main.cu` 中井孔区域设置为

\[
\rho=1000,
\qquad
\rho_f=1000,
\qquad
C_2=1000,
\qquad
C_1=0.
\]

因此

\[
Q=C_2-\frac{\rho_f^2}{\rho}
=1000-\frac{1000^2}{1000}=0.
\]

这说明井孔纯流体区域中相对流体速度方程退化。代码中已经通过

```c
if (C1x[offset] == 0.0f) {
    vwx[offset] = 0.0f;
}
```

将该区域的 \(\mathbf v_\omega\) 置零。因此在计算 \(Q^{-1}\)、\(A_\omega\)、\(B_p\)、\(B_\tau\) 时，必须跳过该区域。

推荐判断条件为：

```c
Q = C2 - rhof * rhof * rho_inv;
if (C1 == 0.0f || fabsf(Q) < 1.0e-12f) {
    vw_tilde = 0.0f;
} else {
    // use scaled coefficients
}
```

---

## 17. 推荐的代码改写流程

### Step 1：预处理阶段计算全局缩放因子

在 CPU 端模型参数生成完成后，扫描 `rho_ext, rhof_ext, C1_ext, C2_ext, HH_ext, H2u_ext, C_ext, M_ext, mu_ext`，计算：

\[
K_{\max}=\max(|HH|,|H2u|,|C|,|M|,|\mu|),
\]

\[
R_{\max}=\max\left(|\rho^{-1}|,|Q^{-1}|,\left|\frac{\rho_f}{\rho Q}\right|\right),
\]

其中 \(Q=C_2-\rho_f^2\rho^{-1}\)，并跳过 \(C_1=0\) 或 \(|Q|\) 很小的纯流体区域。

然后计算

\[
e_r^B=\left(\Delta t^2K_{\max}R_{\max}\right)^{-1/2},
\]

\[
e_m^B=-\log_2(e_r^B\Delta tK_{\max}).
\]

当前 `main.cu` 对应推荐值：

```c
const int   em_int = -23;
const float scale_m = 1.1920928955e-7f;   // 2^(-23)
const float inv_scale_m = 8388608.0f;     // 2^(23)
const int   es_int = 0;
```

### Step 2：预先生成缩放后的参数数组

将原来的参数数组改为：

```c
rho_inv_tilde = inv_scale_m * DT * rho_inv;
C_tilde       = scale_m * DT * C;
M_tilde       = scale_m * DT * M;
HH_tilde      = scale_m * DT * HH;
H2u_tilde     = scale_m * DT * H2u;
mu_tilde      = scale_m * DT * mu;
```

对于相对流体速度方程，推荐预先生成：

```c
Q = C2 - rhof * rhof * rho_inv;
Aw = (Q - 0.5f * C1 * DT) / (Q + 0.5f * C1 * DT);
Bp = inv_scale_m * DT / (Q + 0.5f * C1 * DT);
Bt = inv_scale_m * DT * rhof * rho_inv / (Q + 0.5f * C1 * DT);
```

然后在 kernel 中直接使用 `Aw, Bp, Bt`，避免重复计算和大数除法。

### Step 3：kernel 中不再重复乘以 `DT`

压力和应力更新中，所有模量型系数已经包含 \(\Delta t\)，因此不再写

```c
DT * C_ext[offset]
DT * M_ext[offset]
DT * HH_ext[offset]
DT * muxy[offset]
```

而是直接使用缩放后的

```c
C_tilde[offset]
M_tilde[offset]
HH_tilde[offset]
muxy_tilde[offset]
```

### Step 4：输出时恢复物理量

如果输出速度，需要恢复：

\[
\mathbf v_u=2^{e_m^B-e_s^B}\tilde{\mathbf v}_u.
\]

当前 \(e_m^B=-23, e_s^B=0\)，所以

\[
\mathbf v_u=2^{-23}\tilde{\mathbf v}_u.
\]

代码中：

```c
vux_physical = scale_m * vux_tilde;
vwx_physical = scale_m * vwx_tilde;
```

如果输出应力和压力，由于当前 \(e_s^B=0\)，则

```c
txx_physical = txx_tilde;
ss_physical  = ss_tilde;
```

如果未来使用非零 \(e_s^B\)，则应恢复为

```c
stress_physical = powf(2.0f, -es_int) * stress_tilde;
pressure_physical = powf(2.0f, -es_int) * pressure_tilde;
velocity_physical = powf(2.0f, em_int - es_int) * velocity_tilde;
```

---

## 18. 推荐写入论文方法部分的简洁表述

可以在论文方法中将该方法概括为：

> To avoid the manual tuning of multiple linear scaling coefficients in the original BEHP formulation, we introduce an exponent-based scaling strategy for the poroelastic wave equations. The solid and relative-fluid velocities are scaled by \(2^{e_s^B-e_m^B}\), whereas the pore pressure and stress tensor are scaled by \(2^{e_s^B}\). The exponent \(e_m^B\) is automatically determined from the maximum modulus-like coefficient and the maximum inertia-like coefficient in the governing equations, thereby balancing the magnitudes of the scaled stiffness and density-related terms within the FP16 numerical space. This strategy converts the original poroelastic equations into a dimensionless update form in which all major coefficients remain within the representable and numerically stable range of FP16, while avoiding case-by-case manual adjustment of scaling parameters.

---

## 19. 当前数值结果总结

对于当前 `main.cu`，推荐使用：

\[
\boxed{e_m^B=-23}
\]

\[
\boxed{2^{e_m^B}=1.1920928955\times10^{-7}}
\]

\[
\boxed{2^{-e_m^B}=8.388608\times10^6}
\]

\[
\boxed{e_s^B=0}
\]

关键缩放后系数为：

| 系数 | 数值 |
|---|---:|
| \(\tilde\rho^{-1}\) | \(3.0381276459\times10^{-3}\) |
| \(\tilde Q^{-1}\) | \(2.0351047368\times10^{-4}\) |
| \(\tilde R_Q\) | \(8.1895562847\times10^{-5}\) |
| \(\tilde C_1^Q\) | \(1.2130169492\times10^{-2}\) |
| \(\tilde H\) | \(4.1182637215\times10^{-3}\) |
| \(\tilde H_{2\mu}\) | \(3.8749337196\times10^{-4}\) |
| \(\tilde C\) | \(2.4139881134\times10^{-4}\) |
| \(\tilde M\) | \(2.4139881134\times10^{-3}\) |
| \(\tilde\mu\) | \(1.8653851748\times10^{-3}\) |
| \(A_\omega\) | \(0.9879429575\) |
| \(B_p\) | \(2.0228360646\times10^{-4}\) |
| \(B_\tau\) | \(8.1401853706\times10^{-5}\) |
| \(\rho_f\rho^{-1}\) | \(0.4024144869\) |

这些系数整体位于 \(10^{-5}\sim10^{-2}\) 量级，比原始方程中 \(10^{10}\) 量级的模量项和 \(10^{-4}\) 量级的密度倒数项更加适合 FP16 存储和计算。

---

## 20. 重要注意事项

1. 井孔纯流体区域中 \(Q=0\)，不能计算 \(Q^{-1}\)。必须沿用当前代码中的逻辑，将 \(\mathbf v_\omega\) 置零。

2. 当前推荐 \(e_s^B=0\) 是因为 `main.cu` 的震源是直接加到应力上的无量纲 Ricker 波形。如果后续震源改为物理应力源或压力源，应重新计算 \(e_s^B\)。

3. 如果采用 FP16 存储 + FP32 计算，则上述缩放已经可以显著降低存储压力，并避免大参数进入 FP16。若进一步使用 FP16 直接计算，应额外检查每一步中乘法、加法和 PML 记忆变量是否发生下溢或精度损失。

4. 本文给出的 \(e_m^B=-23\) 是基于当前 `main.cu` 的固定模型参数。如果后续模型参数、孔隙度、渗透率、流体密度、时间步长发生改变，应重新扫描模型参数并自动计算 \(K_{\max}\)、\(R_{\max}\) 和 \(e_m^B\)。
