import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.inf)
pi = math.pi
def getz(option):
    switch_dict = {
        1: 0,
        2: 50,
        3: 200,
        4: 250
    }
    return switch_dict.get(option, "Invalid option")

def fit(points):
    # 计算点的重心
    center = np.mean(points, axis=0)

    # 使用PCA计算主要方向
    pca = PCA(n_components=1)
    pca.fit(points - center)
    direction = pca.components_[0]

    #plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）字体
    #plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 打印重心和方向向量
    #print(f"重心: {center}")
    #print(f"方向向量: {direction}")

    # 可视化拟合的直线
    #t = np.linspace(-10, 10, 100)
    #line_points = center + t[:, np.newaxis] * direction

    # 计算每个点到拟合直线的垂直距离
    # 通过点到直线的距离公式计算
    def point_line_distance(point, line_point, line_direction):
        line_direction = line_direction / np.linalg.norm(line_direction)  # 归一化方向向量
        vector = point - line_point
        distance = np.linalg.norm(np.cross(vector, line_direction))  # 点到直线的距离
        return distance

    # 计算所有点到拟合直线的距离
    distances = np.array([point_line_distance(p, center, direction) for p in points])

    # 计算总平方误差 (SSE)
    sse = np.sum(distances ** 2)

    # 计算总平方总误差 (SST)
    # SST 是点到重心的距离的平方和
    center_distances = np.linalg.norm(points - center, axis=1)
    sst = np.sum(center_distances ** 2)

    # 计算 R²
    r_squared = 1 - (sse / sst)

    '''
    # 打印 R² 值
    print(f"拟合的 R² 值: {R_squared}")

    # 绘制原始点和拟合直线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='原始点')
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='b', label='拟合直线')
    ax.legend()
    plt.show()
    '''
    return r_squared

data = np.loadtxt('output.txt')
data_list = np.array([int(x) for x in data])
zid = np.array([int(x/1e8) for x in data_list])
z = np.array([getz(x) for x in zid])
yid = np.array([int(x % 1e4) for x in data_list])
x0 = np.array([int(x/1e4) for x in data_list])
xid = np.array([int(x % 1e4) for x in x0])
x = np.array([round((2001-x) * 0.025,3) for x in xid])
y = np.array([round((2001-x) * 0.025,3) for x in yid])
hp = np.array([[x,y,z] for x, y, z in zip(x, y, z) ])
hp1 = np.array([x for x in hp if x[2] == 0])
hp2 = np.array([x for x in hp if x[2] == 50])
hp3 = np.array([x for x in hp if x[2] == 200])
hp4 = np.array([x for x in hp if x[2] == 250])
n1 = hp1.shape[0]
n2 = hp2.shape[0]
n3 = hp3.shape[0]
n4 = hp4.shape[0]
print(n1,n2,n3,n4)
vec1 = np.empty((0,3))
vec2 = np.empty((0,3))
match = np.empty((0,4))

for x in hp1:
    for y in hp2:
        new1 = (y-x)/np.linalg.norm(y-x)
        vec1 = np.vstack([vec1,new1])
for x in hp3:
    for y in hp4:
        new2= (y-x)/np.linalg.norm(y-x)
        vec2= np.vstack([vec2,new2])

#match vector between vec1 and vec2
for i in range(n1*n2):
    nx = int(i/n2)
    ny = i-nx*n2
    x2 = 150/vec1[i][2] * vec1[i][0] + hp2[ny][0]
    y2 = 150/vec1[i][2] * vec1[i][1] + hp2[ny][1]
    for j in range(n3):
        d = np.linalg.norm(hp3[j]-np.array([x2,y2,200]))
        if d < 8:
            k = j*n4
            while k > j*n4-1:
                if k == j*n4+n4:break
                cos0 = np.dot(vec1[i],vec2[k])
                cos1 = np.dot(vec1[i],np.array([0,0,1]))
                if cos0>1 or cos0<-1:cos0 = int(cos0)
                angle0 = np.arccos(cos0)
                angle1 = np.arccos(cos1)
                if angle0 < pi/36 and angle1 < pi/12:
                    new3 = np.array([nx,ny,j,k-j*n4])
                    match = np.vstack([match,new3])
                k += 1
#print(match) #first filter

m1 = np.array([x[0] for x in match])
uni0 = np.unique(m1, return_counts=True)
uni1 = uni0[0]
if uni1.size == n1: print("all hits in hp1 was used")
#'''
#filter 0
rep = np.empty((0,2))
num = match.shape[0]
de = []
for i in range(num):
    for j in range(num):
        if j<=i:continue
        dma = (match[i] - match[j])
        if dma[0] * dma[1] * dma[2] * dma[3] == 0:
            rep = np.vstack([rep,[i,j]])

for i in range(rep.shape[0]):
    p1 = np.empty((0,3))
    p2 = np.empty((0,3))
    p1 = np.vstack([p1,hp1[int(match[int(rep[i][0])][0])]])
    p1 = np.vstack([p1,hp2[int(match[int(rep[i][0])][1])]])
    p1 = np.vstack([p1,hp3[int(match[int(rep[i][0])][2])]])
    p1 = np.vstack([p1,hp4[int(match[int(rep[i][0])][3])]])
    p2 = np.vstack([p2,hp1[int(match[int(rep[i][1])][0])]])
    p2 = np.vstack([p2,hp2[int(match[int(rep[i][1])][1])]])
    p2 = np.vstack([p2,hp3[int(match[int(rep[i][1])][2])]])
    p2 = np.vstack([p2,hp4[int(match[int(rep[i][1])][3])]])
    if fit(p1) < fit(p2):
        de.append(int(rep[i][0]))
    else:
        de.append(int(rep[i][1]))
de = np.array([int(x) for x in de])
uni = np.unique(de, return_counts=True)
de = uni[0]
de = de.tolist()
num0 = np.arange(0,num)
print(de)
# correction
y0 = [0]
while len(y0) != 0:
    num1 = np.delete(num0, de)
    y0 = []
    for y in de:
        ii = 0
        for x in num1:
            dma0 = match[x] - match[y]
            if dma0[0] * dma0[1] * dma0[2] * dma0[3] != 0:
                ii += 1
        if ii == num1.size:
            y0.append(y)

    rep0 = np.empty((0, 2))
    for i in range(len(y0)):
        for j in range(len(y0)):
            if j <= i: continue
            dma1 = match[y0[i]] - match[y0[j]]
            if dma1[0] * dma1[1] * dma1[2] * dma1[3] == 0:
                rep0 = np.vstack([rep0, [y0[i], y0[j]]])

    for i in range(rep0.shape[0]):
        p1 = np.empty((0, 3))
        p2 = np.empty((0, 3))
        p1 = np.vstack([p1, hp1[int(match[int(rep0[i][0])][0])]])
        p1 = np.vstack([p1, hp2[int(match[int(rep0[i][0])][1])]])
        p1 = np.vstack([p1, hp3[int(match[int(rep0[i][0])][2])]])
        p1 = np.vstack([p1, hp4[int(match[int(rep0[i][0])][3])]])
        p2 = np.vstack([p2, hp1[int(match[int(rep0[i][1])][0])]])
        p2 = np.vstack([p2, hp2[int(match[int(rep0[i][1])][1])]])
        p2 = np.vstack([p2, hp3[int(match[int(rep0[i][1])][2])]])
        p2 = np.vstack([p2, hp4[int(match[int(rep0[i][1])][3])]])
        mv0 = int(rep0[i][0])
        mv1 = int(rep0[i][1])
        if fit(p1) < fit(p2):
            if mv0 in y0:
                y0.remove(mv0)
        else:
            if mv1 in y0:
                y0.remove(mv1)
    print(y0)
    for x in y0: de.remove(x)

print(de)
match = np.delete(match,de,axis=0)
#match = np.vstack([match,np.array([2,2,2,2])])
#print(match) #second filter
#'''
points0 = np.empty((0,3))
num = match.shape[0]
for i in range(num):
    points0 = np.vstack([points0,hp1[int(match[i][0])]])
    points0 = np.vstack([points0,hp2[int(match[i][1])]])
    points0 = np.vstack([points0,hp3[int(match[i][2])]])
    points0 = np.vstack([points0,hp4[int(match[i][3])]])
#need more filter?

nre = match.shape[0]
nto = 40
eps = nre/nto

print("rebuild efficiency：", eps)

#real points position
#real = """"""
#file = real.strip().split('\n')
coordinates = []
with open('particle_info.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            try:
                coos = tuple(map(float, line.strip('()').split(',')))
                coordinates.append(coos)
            except ValueError:continue
coordinates = [sublist for sublist in coordinates if len(sublist) >= 3]
real_np = np.array(coordinates)

#rebuild trajectory plot and real points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(num):
    x, y, z = points0[4*i:4*i+4, 0], points0[4*i:4*i+4, 1], points0[4*i:4*i+4, 2]
    ax.plot(x, y, z)

x0 = real_np[:,0]
y0 = real_np[:,1]
z0 = real_np[:,2]
ax.scatter(x0, y0, z0)

ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

