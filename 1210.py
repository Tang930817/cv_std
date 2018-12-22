# 求过两点方程,返回斜率k和截距b
def solve_fun(v1,v2):
    k = (v2[1]-v1[1])/(v2[0]-v1[0])
    b = v1[1]-v1[0]*(v2[1]-v1[1])/(v2[0]-v1[0]) 
    return k,b

# 解两条直线交点
def solve_cro(k1,b1,k2,b2):
    x = (b2-b1)/(k1-k2)
    y = k1*x + b1
    return x,y

# 求距离
def cal_dis(v1,v2):
     return ((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)**0.5

# 像素转图上坐标
def piex_trans(u,v):
    u =  -1920/2+u 
    v =  1080/2-v
    return u,v

# 输入初始四个角点
a = piex_trans(1254,227)
b = piex_trans(1135,284)
c = piex_trans(1600,352)
d = piex_trans(1661,282)
# per_piex = (1/2.7)/2202.90717

# 求ad,bc交点，第一个灭点vu
k1,b1 = solve_fun(a,d)
k2,b2 = solve_fun(b,c)
vu = solve_cro(k1,b1,k2,b2)

# 求ab,dc交点，第一个灭点vf
k3,b3 = solve_fun(a,b)
k4,b4 = solve_fun(d,c)
vf = solve_cro(k3,b3,k4,b4)

# 求垂足
kf,bf = solve_fun(vu,vf)
kp = -1/kf
bp = 0
ver_foot = solve_cro(kf,bf,kp,bp)

# 求三个距离l0,l1,l2
p0 = (0,0)
l0 =  cal_dis(p0,ver_foot)
l1 = cal_dis(vf,ver_foot)
l2 = cal_dis(vu,ver_foot)

# 求焦距
f = (l1*l2-l0*l0)**0.5
print(f)


