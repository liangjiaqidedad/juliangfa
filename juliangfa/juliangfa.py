import numpy as np #用于矩阵计算
import matplotlib.pyplot as plt #用于作图
from matplotlib import rcParams


#超参数设置：
l=0.25
n=21     #网格数
a=0.001 #导线半径
f=300e6 #频率
c=3e8 #光速
Z_0=377 #特征阻抗
theta_deg=np.linspace(0,180,181) #角度范围
theta_rad=theta_deg*np.pi/180 #角度范围转换为弧度
phi=0
wavelength=c/f #波长是1m
k=2*np.pi/wavelength #波数
M=n-1 #阻抗矩阵维度
#-------第一步，计算Zm和Zn--------
Zn=np.linspace(-l,l,n)
Zm=(Zn[:-1]+Zn[1:])/2 #计算中点值     
delta_z=2*l/(n-1) #计算区间段长度
print(f"Zn={Zn}")
print(f"\nZm={Zm}")
print(f"\ndelta_z={delta_z}")

#-------第二步，计算阻抗矩阵Z--------
Z=np.zeros((M,M),dtype=complex)
for i in range(M):
    for j in range(M):
        if j==0:#根据公式第一列的元素是由bm1决定
            Z[i,j]=np.cos(k*Zm[i])
        else:   
            if i==j:#对角元素的泰勒展开处理
                r=np.sqrt(delta_z**2+4*a**2)
                Z_1=(1/(4*np.pi))*np.log((r+delta_z)/(r-delta_z))
                Z_2=(-1j*k/(4*np.pi))*delta_z
                Z_3=(1/(8*np.pi))*(delta_z/2*r+a**2*np.log((r+delta_z)/(r-delta_z)))*k**2/2
                Z[i,j]=Z_1+Z_2+Z_3
            else:
                R=np.sqrt(a**2+(Zm[i]-Zm[j])**2)
                Z[i,j]=np.exp(-1j*k*R)/(4*np.pi*R)*delta_z
                Z_abs=np.abs(Z[i,j])
print(f"\nZ={Z}\n\n")
print(Z[:,0])

#-------第三步，构造矩阵b-----------
b=np.zeros((M,1),dtype=complex)
for m in range(M):
    b[m,0]=-1j*np.sin(k*abs(Zm[m]))/(2*Z_0)
print(f"\nb={b}")

#-------第四步，求解电流矩阵I-----------
I=np.linalg.solve(Z,b)
B=-I[0,0]
I[0,0]=0
print(f"\nB={B}")
print(f"\nI={I}")

#-------第五步，计算馈入电压和输入阻抗------
center = (n+1)//2
V_in = 1  # 馈电点电压（来自激励）
I_in = I[center, 0]  # 馈电点电流
Z_in = V_in / I_in  # 输入阻抗
print(f"\nZ_in={Z_in}")

#------第六 计算远场进场--------
R_near=10
R_far=100
E_near=np.zeros((len(theta_rad),1),dtype=complex)
E_far=np.zeros((len(theta_rad),1),dtype=complex)
for i in range(len(theta_rad)):
    integral=0
    for m in range(M):
        integral += I[m,0] * np.exp(1j*k*Zm[m]*np.cos(theta_rad[i])) * delta_z  # 注意：是 1j，不是 j
    # 近场电场
    E_near[i] = (1j*k*Z_0/(4*np.pi*R_near)) * np.exp(-1j*k*R_near) * np.sin(theta_rad[i]) * integral
    # 远场电场
    E_far[i] = (1j*k*Z_0/(4*np.pi*R_far)) * np.exp(-1j*k*R_far) * np.sin(theta_rad[i]) * integral

# 计算幅值
E_near_mag = np.abs(E_near)
E_far_mag = np.abs(E_far)

# 归一化（相对于最大值，转换为dB）
E_near_normalized = E_near_mag / np.max(E_near_mag)
E_far_normalized = E_far_mag / np.max(E_far_mag)
E_near_dB = 20 * np.log10(E_near_normalized + 1e-10)  # 加小量避免log(0)
E_far_dB = 20 * np.log10(E_far_normalized + 1e-10)

print(f"\n进场电场幅值:{E_near_dB}")
print(f"\远场电场幅值:{E_far_dB}")
print(f"\n最大近场幅值: {np.max(E_near_mag):.6e} V/m")
print(f"最大远场幅值: {np.max(E_far_mag):.6e} V/m")
print(f"幅值比（近场/远场）: {np.max(E_near_mag)/np.max(E_far_mag):.2e}")

#------第七 计算方向性系数--------
print("\n" + "="*60)
print("方向性系数计算:")

# 1. 计算辐射强度 U(θ) = r² |E(θ)|² / (2Z₀)
U_far = R_far**2 * E_far_mag**2 / (2 * Z_0)  # 辐射强度（W/sr）

# 2. 找到最大辐射强度
U_max = np.max(U_far)
max_direction_idx = np.argmax(U_far)
max_direction_deg = theta_deg[max_direction_idx]

print(f"最大辐射方向: θ = {max_direction_deg:.1f}°")
print(f"最大辐射强度: U_max = {U_max:.6e} W/sr")

# 计算总辐射功率
# P_rad = 2π ∫[0,π] U(θ) sinθ dθ
dtheta = theta_rad[1] - theta_rad[0]  # 角度步长（弧度）
integrand = U_far.flatten() * np.sin(theta_rad)  # 被积函数：U(θ)·sinθ
P_rad = 2 * np.pi * np.trapz(integrand, theta_rad)  # 总辐射功率

print(f"总辐射功率: P_rad = {P_rad:.6e} W")

# 4. 计算方向性系数 D = 4π U_max / P_rad
Directivity = 4 * np.pi * U_max / P_rad
Directivity_dB = 10 * np.log10(Directivity)

print(f"\n方向性系数:")
print(f"  D = {Directivity:.4f} (线性)")
print(f"  D = {Directivity_dB:.2f} dBi")



#------第七步，绘制方向图--------
print("\n" + "="*60)
print("正在生成方向图...")

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 10))

# 1. 电流幅值分布
ax1 = plt.subplot(1, 3, 1)
ax1.plot(Zm, np.abs(I), 'b-o', linewidth=2, markersize=5)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='r', linestyle='--', alpha=0.3, label='中心')
ax1.set_xlabel('位置 z (m)', fontsize=11)
ax1.set_ylabel('电流幅值 |I| (A)', fontsize=11)
ax1.set_title('天线电流分布', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()


# 5. 远场极坐标方向图
ax5 = plt.subplot(1, 3, 2, projection='polar')
E_far_dB_clipped = np.maximum(E_far_dB, -40)
E_far_linear = 10**(E_far_dB_clipped / 20)
ax5.plot(theta_rad, E_far_linear, 'b-', linewidth=2)
ax5.fill(theta_rad, E_far_linear, alpha=0.25)
ax5.set_theta_zero_location('E')  
ax5.set_theta_direction(1)  # 顺时针
ax5.set_title('远场方向图（极坐标）', fontsize=12, fontweight='bold', pad=20)
ax5.grid(True, alpha=0.3)

# 6. 近场极坐标方向图
ax6 = plt.subplot(1, 3, 3, projection='polar')
E_near_dB_clipped = np.maximum(E_near_dB, -40)
E_near_linear = 10**(E_near_dB_clipped / 20)
ax6.plot(theta_rad, E_near_linear, 'g-', linewidth=2)
ax6.fill(theta_rad, E_near_linear, alpha=0.25)
ax6.set_theta_zero_location('E')
ax6.set_theta_direction(1)
ax6.set_title('近场方向图（极坐标）', fontsize=12, fontweight='bold', pad=20)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('antenna_pattern.png', dpi=300, bbox_inches='tight')
print("方向图已保存为 'antenna_pattern.png'")

plt.show()

print("\n" + "="*60)
print("计算完成！")
