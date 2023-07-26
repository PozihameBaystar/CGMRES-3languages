import math
import numpy as np
import time

#移動ロボットクラス
class car:
    def __init__(self):
        self.R = 0.05
        self.T = 0.2
        self.r = self.R/2
        self.rT = self.R/self.T

    def func(self,u,x):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.r*cos_*(u[0][0]+u[1][0])],
                       [self.r*sin_*(u[0][0]+u[1][0])],
                       [self.rT*(u[0][0]-u[1][0])]])
        return dx
    

#コントローラークラス
class controller:
    def __init__(self, car, x_ob):
        #コントローラーのパラメータ
        self.Ts = 0.05 #制御周期
        self.ht = self.Ts 
        self.zeta = 1.0/self.Ts #安定化係数
        self.tf = 1.0 #予測ホライズンの最終値
        self.alpha = 0.5 #予測ホライズンの変化パラメータ
        self.N = 20 #予測ホライズンの分割数
        self.Time = 0.0 #時刻を入れる変数
        self.dt = 0.0 #予測ホライズンの分割幅

        #入力と状態の次元
        self.len_u = 2 #入力の次元
        self.len_x = 3 #状態変数の次元

        #評価関数の重み
        self.Q = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        self.R = np.array([[1, 0],
                           [0, 1]])
        self.S = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        
        #コントローラーの変数
        self.u = np.zeros((self.len_u, 1))
        self.U = np.zeros((self.len_u * self.N, 1))
        self.x = np.zeros((self.len_x, 1))
        self.dU = np.zeros((self.len_u * self.N, 1))

        #入力の制限
        self.umax = np.array([[15],
                              [15]]) #各入力の最大値
        self.umin = np.array([[-15],
                              [-15]]) #各入力の最小値
        
        #目標地点
        self.x_ob = x_ob

        #操縦する車
        self.car = car

    def CGMRES_control(self):
        self.dt = (1-math.exp(-self.alpha*self.Time))*self.tf/self.N
        dx = self.func(self.x, self.u)

        Fux = self.CalcF(self.x + dx*self.ht, self.U + self.dU*self.ht)
        Fx = self.CalcF(self.x + dx*self.ht, self.U)
        F = self.CalcF(self.x, self.U)

        left = (Fux - Fx)/self.ht
        right = -self.zeta*F - (Fx - F)/self.ht
        r0 = right - left

        m = self.len_u*self.N

        Vm = np.zeros((self.len_u*self.N, m+1))
        Vm[:,0:1] = r0/np.linalg.norm(r0)

        Hm = np.zeros((m+1,m))

        for i in range(m):
            Fux = self.CalcF(self.x + dx*self.ht, self.U + Vm[:,i:i+1]*self.ht)
            Av = (Fux - Fx)/self.ht

            for k in range(i+1):
                Hm[k][i] = np.matmul(Av.T,Vm[:,k:k+1])

            temp_vec = np.zeros((self.len_u*self.N, 1))
            for k in range(i+1):
                temp_vec = temp_vec + Hm[k][i]*Vm[:,k:k+1]

            v_hat = Av - temp_vec

            Hm[i+1][i] = np.linalg.norm(v_hat)
            Vm[:,i+1:i+2] = v_hat/Hm[i+1][i]

        e = np.zeros((m+1, 1))
        e[0][0] = 1.0
        gm_ = np.linalg.norm(r0)*e

        UTMat, gm_ = self.ToUTMat(Hm, gm_, m)

        min_y = np.zeros((m, 1))

        for i in range(m):
            min_y[i][0] = (gm_[i][0] - np.matmul(UTMat[i:i+1,:],min_y))/UTMat[i][i]

        dU_new = self.dU + np.matmul(Vm[:,0:m], min_y)

        self.dU = dU_new
        self.U = self.U + self.dU*self.ht
        self.u = self.U[0:2,0:1]


    #コントローラー側の運動方程式
    def func(self, x, u):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.car.r*cos_*(u[0][0]+u[1][0])],
                       [self.car.r*sin_*(u[0][0]+u[1][0])],
                       [self.car.rT*(u[0][0]-u[1][0])]])
        return dx
    
    #F
    def CalcF(self, x, U):
        F = np.zeros((self.len_u*self.N, 1))
        U = U.reshape(self.len_u, self.N, order='F')
        X, B_all = self.Forward(x, U)

        Lambda = self.Backward(X, U)

        for i in range(self.N):
            F[self.len_u*i:self.len_u*(i+1), 0:1] = self.CalcHu(U[:,i:i+1], Lambda[:,i:i+1], B_all[:,self.len_u*i:self.len_u*(i+1)])

        return F
    
    #xの予測計算
    def Forward(self, x, U):
        X = np.zeros((self.len_x, self.N+1))
        B_all = np.zeros((self.len_x, self.len_u*self.N))

        X[:,0:1] = x

        for i in range(1,self.N+1):
            dx, B = self.funcB(X[:,i-1:i], U[:,i-1:i])
            X[:,i:i+1] = X[:,i-1:i] + dx*self.dt
            B_all[:,self.len_u*(i-1):self.len_u*i] = B

        return X, B_all
    
    #随伴変数の計算
    def Backward(self, X, U):
        Lambda = np.zeros((self.len_x, self.N))
        Lambda[:,self.N-1:self.N] = np.matmul(self.S, X[:,self.N:self.N+1]-self.x_ob)

        for i in reversed(range(self.N-1)):
            Lambda[:,i:i+1] = Lambda[:,i+1:i+2] + self.CalcHx(X[:,i+1:i+2], U[:,i+1:i+2], Lambda[:,i+1:i+2])*self.dt
        
        return Lambda
    
    #dH/du
    def CalcHu(self, u, lambd, B):
        dHdu = np.zeros((self.len_u, 1))
        dHdu = np.matmul(self.R, u) + np.matmul(B.T, lambd)
        dHdu[0][0] += 0.15*(2*u[0][0] - self.umax[0][0] - self.umin[0][0])/((u[0][0] - self.umin[0][0])*(self.umax[0][0] - u[0][0]))
        dHdu[1][0] += 0.15*(2*u[1][0] - self.umax[1][0] - self.umin[1][0])/((u[1][0] - self.umin[1][0])*(self.umax[1][0] - u[1][0]))

        return dHdu
    
    #dHdx
    def CalcHx(self, x, u, lambd):
        dHdx = np.zeros((self.len_x, 1))
        dfdx = self.Calcfx(x,u)
        dHdx = np.matmul(self.Q, x-self.x_ob) + np.matmul(dfdx.T, lambd)
        return dHdx
    
    #dfdx
    def Calcfx(self, x, u):
        dfdx = np.array([[0, 0, -self.car.r*math.sin(x[2][0])*(u[0][0]+u[1][0])],
                         [0, 0, self.car.r*math.cos(x[2][0])*(u[0][0]+u[1][0])],
                         [0, 0, 0]])
        return dfdx
    
    #Givens回転
    def ToUTMat(self, Hm, gm, m):
        for i in range(m):
            nu = math.sqrt(Hm[i][i]**2 + Hm[i+1][i]**2)
            c_i = Hm[i][i]/nu
            s_i = Hm[i+1][i]/nu
            Omega = np.eye(m+1)
            Omega[i][i] = c_i
            Omega[i][i+1] = s_i
            Omega[i+1][i] = -s_i
            Omega[i+1][i+1] = c_i

            Hm = np.matmul(Omega, Hm)
            gm = np.matmul(Omega, gm)

        return Hm, gm
    
    #Forward用func
    def funcB(self, x, u):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])

        B = np.array([[self.car.r*cos_, self.car.r*cos_],
                      [self.car.r*sin_, self.car.r*sin_],
                      [self.car.rT, -self.car.rT]])
        
        dx = np.matmul(B, u)

        return dx, B
    

if __name__ == "__main__":
    x_ob = np.array([[3, 2, 0]]).T

    nonholo_car = car()
    CGMRES_cont = controller(nonholo_car, x_ob)

    Time = 0

    start = time.time()
    while Time <= 20:
        print("-------------Position-------------")
        print(CGMRES_cont.x)

        x = CGMRES_cont.x + nonholo_car.func(CGMRES_cont.u, CGMRES_cont.x)*CGMRES_cont.Ts

        CGMRES_cont.Time = Time + CGMRES_cont.Ts

        CGMRES_cont.CGMRES_control()

        Time += CGMRES_cont.Ts

        CGMRES_cont.x = x

    end = time.time()

    loop_time = end - start

    print("計算時間：{}[s]".format(loop_time))