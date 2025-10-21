import numpy as np #用于矩阵计算
import matplotlib.pyplot as plt #用于作图
from matplotlib import rcParams
import pandas as pd #用于导出Excel


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

#------第六 计算远场幅值--------
R_near=10
R_far=100
E_near=np.zeros((len(theta_rad),1),dtype=complex)
E_far=np.zeros((len(theta_rad),1),dtype=complex)
for i in range(len(theta_rad)):
    integral=0
    for m in range(M):
        integral += I[m,0] * np.exp(1j*k*Zm[m]*np.cos(theta_rad[i])) * delta_z  # 注意：是 1j，不是 j
    # 远场1电场
    E_near[i] = (1j*k*Z_0/(4*np.pi*R_near)) * np.exp(-1j*k*R_near) * np.sin(theta_rad[i]) * integral
    # 远场2电场
    E_far[i] = (1j*k*Z_0/(4*np.pi*R_far)) * np.exp(-1j*k*R_far) * np.sin(theta_rad[i]) * integral

# 计算幅值
E_near_mag = np.abs(E_near)
E_far_mag = np.abs(E_far)

# 归一化（相对于最大值，转换为dB）
E_near_normalized = E_near_mag / np.max(E_near_mag)
E_far_normalized = E_far_mag / np.max(E_far_mag)
E_near_dB = 20 * np.log10(E_near_normalized + 1e-10)  # 加小量避免log(0)
E_far_dB = 20 * np.log10(E_far_normalized + 1e-10)

print(f"\n远场1电场幅值:{E_near_dB}")
print(f"\n远场2电场幅值:{E_far_dB}")
print(f"\n最大远场1幅值: {np.max(E_near_mag):.6e} V/m")
print(f"最大远场2幅值: {np.max(E_far_mag):.6e} V/m")
print(f"幅值比（远场1/远场2）: {np.max(E_near_mag)/np.max(E_far_mag):.2e}")

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
ax1 = plt.subplot(1, 4, 1)
ax1.plot(Zm, np.abs(I), 'b-o', linewidth=2, markersize=5)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='r', linestyle='--', alpha=0.3, label='中心')
ax1.set_xlabel('位置 z (m)', fontsize=11)
ax1.set_ylabel('电流幅值 |I| (A)', fontsize=11)
ax1.set_title('天线电流分布', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2=plt.subplot(1,4,2)
ax2.plot(theta_rad,np.abs(E_far),'b-o',linewidth=2,markersize=5)
ax2.plot(theta_rad,np.abs(E_near),'g-o',linewidth=2,markersize=5)
ax2.set_xlabel('角度 θ (°)', fontsize=11)
ax2.set_ylabel('场强 |E| (V/m)', fontsize=11)
ax2.set_title('远场1和远场2场强分布', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()



# 5. 远场极坐标方向图
ax3 = plt.subplot(1, 4, 3, projection='polar')
E_far_dB_clipped = np.maximum(E_far_dB, -40)
E_far_linear = 10**(E_far_dB_clipped / 20)
ax3.plot(theta_rad, E_far_linear, 'b-', linewidth=2)
ax3.fill(theta_rad, E_far_linear, alpha=0.25)
ax3.set_theta_zero_location('E')  
ax3.set_theta_direction(1)  # 顺时针
ax3.set_title('远场1方向图（极坐标）', fontsize=12, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3)

# 6. 近场极坐标方向图
ax4 = plt.subplot(1, 4, 4, projection='polar')
E_near_dB_clipped = np.maximum(E_near_dB, -40)
E_near_linear = 10**(E_near_dB_clipped / 20)
ax4.plot(theta_rad, E_near_linear, 'g-', linewidth=2)
ax4.fill(theta_rad, E_near_linear, alpha=0.25)
ax4.set_theta_zero_location('E')
ax4.set_theta_direction(1)
ax4.set_title('远场2方向图（极坐标）', fontsize=12, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('antenna_pattern.png', dpi=300, bbox_inches='tight')
print("方向图已保存为 'antenna_pattern.png'")

plt.show()

print("\n" + "="*60)
print("计算完成！")

#------第八步，导出数据到Excel--------
print("\n" + "="*60)
print("正在导出数据到Excel...")

# 创建Excel写入器
with pd.ExcelWriter('天线计算结果.xlsx', engine='openpyxl') as writer:
    
    # 1. 导出阻抗矩阵Z（实部和虚部分开）
    Z_real = np.real(Z)
    Z_imag = np.imag(Z)
    
    # 创建带标题的阻抗矩阵实部
    df_Z_real_title = pd.DataFrame([['阻抗矩阵 Z 的实部 (Ohm)', '', '', '', '', '', '', '', '', '']])
    df_Z_real_title.to_excel(writer, sheet_name='1-阻抗矩阵Z', index=False, header=False, startrow=0)
    
    df_Z_real = pd.DataFrame(Z_real, 
                             index=[f'行{i+1}' for i in range(M)],
                             columns=[f'列{j+1}' for j in range(M)])
    df_Z_real.to_excel(writer, sheet_name='1-阻抗矩阵Z', startrow=2)
    
    # 创建带标题的阻抗矩阵虚部
    df_Z_imag_title = pd.DataFrame([['', ''], ['阻抗矩阵 Z 的虚部 (Ohm)', '']])
    df_Z_imag_title.to_excel(writer, sheet_name='1-阻抗矩阵Z', index=False, header=False, startrow=M+5)
    
    df_Z_imag = pd.DataFrame(Z_imag,
                             index=[f'行{i+1}' for i in range(M)],
                             columns=[f'列{j+1}' for j in range(M)])
    df_Z_imag.to_excel(writer, sheet_name='1-阻抗矩阵Z', startrow=M+7)
    
    # 2. 导出B值和电流矩阵I
    I_real = np.real(I).flatten()
    I_imag = np.imag(I).flatten()
    I_magnitude = np.abs(I).flatten()
    I_phase = np.angle(I, deg=True).flatten()
    
    # B值标题
    df_B_title = pd.DataFrame([['激励系数 B 值']])
    df_B_title.to_excel(writer, sheet_name='2-电流分布', index=False, header=False, startrow=0)
    
    df_I = pd.DataFrame({
        'B值_实部': [np.real(B)],
        'B值_虚部': [np.imag(B)],
        'B值_幅值': [np.abs(B)],
        'B值_相位(度)': [np.angle(B, deg=True)]
    })
    df_I.to_excel(writer, sheet_name='2-电流分布', index=False, startrow=1)
    
    # 电流分布标题
    df_current_title = pd.DataFrame([['', ''], ['天线电流分布 I', '']])
    df_current_title.to_excel(writer, sheet_name='2-电流分布', index=False, header=False, startrow=4)
    
    df_current = pd.DataFrame({
        '位置Zm (m)': Zm,
        '电流_实部 (A)': I_real,
        '电流_虚部 (A)': I_imag,
        '电流_幅值 (A)': I_magnitude,
        '电流_相位 (度)': I_phase
    })
    df_current.to_excel(writer, sheet_name='2-电流分布', startrow=6, index=False)
    
    # 3. 导出b矩阵
    b_real = np.real(b).flatten()
    b_imag = np.imag(b).flatten()
    b_magnitude = np.abs(b).flatten()
    
    # b矩阵标题
    df_b_title = pd.DataFrame([['激励向量 b （右端项矩阵）']])
    df_b_title.to_excel(writer, sheet_name='3-激励向量b', index=False, header=False, startrow=0)
    
    df_b = pd.DataFrame({
        '位置Zm (m)': Zm,
        'b_实部': b_real,
        'b_虚部': b_imag,
        'b_幅值': b_magnitude
    })
    df_b.to_excel(writer, sheet_name='3-激励向量b', index=False, startrow=2)
    
    # 4. 导出电场数据
    E_near_real = np.real(E_near).flatten()
    E_near_imag = np.imag(E_near).flatten()
    E_far_real = np.real(E_far).flatten()
    E_far_imag = np.imag(E_far).flatten()
    
    # 电场数据标题
    df_E_title = pd.DataFrame([['天线辐射电场分布 (近场: R=10m, 远场: R=100m)']])
    df_E_title.to_excel(writer, sheet_name='4-电场分布', index=False, header=False, startrow=0)
    
    df_E = pd.DataFrame({
        '角度 (度)': theta_deg,
        '角度 (弧度)': theta_rad,
        '近场_实部 (V/m)': E_near_real,
        '近场_虚部 (V/m)': E_near_imag,
        '近场_幅值 (V/m)': E_near_mag.flatten(),
        '近场_归一化dB': E_near_dB.flatten(),
        '远场_实部 (V/m)': E_far_real,
        '远场_虚部 (V/m)': E_far_imag,
        '远场_幅值 (V/m)': E_far_mag.flatten(),
        '远场_归一化dB': E_far_dB.flatten()
    })
    df_E.to_excel(writer, sheet_name='4-电场分布', index=False, startrow=2)
    
    # 5. 导出其他参数
    df_params_title = pd.DataFrame([['天线计算参数和结果汇总']])
    df_params_title.to_excel(writer, sheet_name='5-计算参数', index=False, header=False, startrow=0)
    
    df_params = pd.DataFrame({
        '参数名称': ['天线长度l (m)', '网格数n', '导线半径a (m)', '频率f (Hz)', 
                   '波长λ (m)', '波数k (rad/m)', '特征阻抗Z0 (Ω)', 
                   '输入阻抗Zin_实部 (Ω)', '输入阻抗Zin_虚部 (Ω)', '输入阻抗Zin_幅值 (Ω)',
                   '方向性系数D', '方向性系数D (dBi)', '最大辐射方向 (度)'],
        '数值': [l, n, a, f, wavelength, k, Z_0, 
                np.real(Z_in), np.imag(Z_in), np.abs(Z_in),
                Directivity, Directivity_dB, max_direction_deg]
    })
    df_params.to_excel(writer, sheet_name='5-计算参数', index=False, startrow=2)

print("数据已成功导出到 '天线计算结果.xlsx'")
print("包含以下工作表:")
print("  - 1-阻抗矩阵Z (包含实部和虚部)")
print("  - 2-电流分布 (包含B值和电流I)")
print("  - 3-激励向量b")
print("  - 4-电场分布 (包含近场和远场)")
print("  - 5-计算参数")
