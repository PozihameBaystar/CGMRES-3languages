#include <iostream>
#include <math.h>
#include <time.h>
#include "Eigen/Dense"


/*非ホロノミック移動ロボットクラス*/
class car{
    public:
    /*車のサイズに関するパラメータ*/
    double R = 0.05;
    double T = 0.2;
    double r = R/2.0;
    double rT = R/T;

    /*車側の運動方程式（同じもの）*/
    Eigen::VectorXd func(const Eigen::Vector2d &u, const Eigen::Vector3d &x){
        Eigen::VectorXd dx(3);
        double cos_ = cos(x(2));
        double sin_ = sin(x(2));
        dx(0) = r*cos_*(u(0)+u(1));
        dx(1) = r*sin_*(u(0)+u(1));
        dx(2) = rT*(u(0)-u(1));
        return dx;
    }
};


/*コントローラークラス*/
class controller{
    public:
    /*コントローラーのパラメータ*/
    double Ts = 0.05; /*制御周期*/
    double ht = Ts;
    double zeta = 1.0/Ts; /*安定化係数*/
    double tf = 1.0; /*予測ホライズンの最終値*/
    double alpha = 0.5; /*予測ホライズンの変化のパラメータ*/
    int N = 20; /*予測ホライズンの分割数*/
    double Time = 0.0; /*時刻を入れる変数*/
    double dt = 0.0; /*予測ホライズンの分割幅*/

    /*入力と状態変数のサイズ*/
    int len_u = 2;
    int len_x = 3;

    /*評価関数中の重み*/
    Eigen::MatrixXd Q; /*途中の状態の重み*/
    Eigen::MatrixXd R; /*入力の重み*/
    Eigen::MatrixXd S; /*終端状態の重み*/
    
    /*コントローラーの変数及び関数値*/
    Eigen::VectorXd u; /*入力*/
    Eigen::VectorXd U; /*予測ホライズン内の全ての入力*/
    Eigen::VectorXd x; /*状態変数*/
    Eigen::VectorXd dU; /*予測ホライズン内の全ての入力の微分*/

    /*入力の制限*/
    Eigen::VectorXd umax = Eigen::VectorXd::Zero(len_u); /*入力最大値*/
    Eigen::VectorXd umin = Eigen::VectorXd::Zero(len_u); /*入力最小値*/

    /*目標地点*/
    Eigen::VectorXd x_ob;

    /*操縦する車*/
    car car_ob;

    controller(car car_ob_, Eigen::VectorXd x_ob_){
        Q = 100*Eigen::MatrixXd::Identity(len_x,len_x);
        Q(2,2) = 0.0;
        R = Eigen::MatrixXd::Identity(len_u,len_u);
        S = 100*Eigen::MatrixXd::Identity(len_x,len_x);
        S(2,2) = 0.0;
        u = Eigen::VectorXd::Zero(len_u);
        U = Eigen::VectorXd::Zero(len_u*N);
        x = Eigen::VectorXd::Zero(len_x);
        dU = Eigen::VectorXd::Zero(len_u*N);
        umax(0) = 15.0;
        umax(1) = 15.0;
        umin(0) = -15.0;
        umin(1) = -15.0;
        car_ob = car_ob_;
        x_ob = x_ob_;
    }

    void CGMRES_control(){
        dt = (1.0-exp(-alpha*Time))*tf/N; /*予測ホライズンの分割幅を更新*/
        Eigen::VectorXd dx = func(x,u);

        Eigen::VectorXd Fux = CalcF(x+dx*ht,U+ht*dU);
        Eigen::VectorXd Fx = CalcF(x+dx*ht,U);
        Eigen::VectorXd F = CalcF(x,U);

        Eigen::VectorXd left = (Fux - Fx)/ht;
        Eigen::VectorXd right = -zeta*F - (Fx - F)/ht;
        Eigen::VectorXd r0 = right - left;

        int m = len_u*N;

        Eigen::MatrixXd Vm(len_u*N,m+1);
        Vm.col(0) = r0/r0.norm();

        Eigen::MatrixXd Hm = Eigen::MatrixXd::Zero(m+1,m);

        for(int i=0; i<m; i++){
            Fux = CalcF(x+dx*ht,U+Vm.col(i)*ht);
            Eigen::VectorXd Av = (Fux - Fx)/ht;

            for(int k=0; k<=i; k++){
                Hm(k,i) = Av.dot(Vm.col(k));
            }

            Eigen::VectorXd temp_vec = Eigen::VectorXd::Zero(len_u*N);
            for(int k=0; k<=i; k++){
                temp_vec = temp_vec + Hm(k,i)*Vm.col(k);
            }

            Eigen::VectorXd v_hat = Av - temp_vec;

            Hm(i+1,i) = v_hat.norm();
            Vm.col(i+1) = v_hat/Hm(i+1,i);
        }

        Eigen::VectorXd e = Eigen::VectorXd::Zero(m+1);
        e(0) = 1.0;
        Eigen::VectorXd gm_ = r0.norm()*e;

        /*std::pair<Eigen::MatrixXd, Eigen::MatrixXd> UTgm =*/
        ToUTMat(Hm,gm_,m);
        Eigen::MatrixXd UTMat = Hm; /*UTgm.first;*/
        /*gm_ = UTgm.second;*/

        Eigen::VectorXd min_y = Eigen::VectorXd::Zero(m);

        for(int i=m-1; i>=0; i--){
            min_y(i) = (gm_(i) - UTMat.row(i)*min_y)/UTMat(i,i);
        }

        Eigen::VectorXd dU_new = dU + Vm.block(0,0,len_u*N,m)*min_y;

        dU = dU_new;
        U = U + dU*ht;
        u = U.segment(0,2);
    }

    /*コントローラーに与えられた運動方程式（モデル誤差無しと仮定しているので車側と同じ）*/
    Eigen::VectorXd func(const Eigen::Vector3d &x, const Eigen::Vector2d &u){
        Eigen::VectorXd dx(3);
        double cos_ = cos(x(2));
        double sin_ = sin(x(2));
        dx(0) = car_ob.r*cos_*(u(0)+u(1));
        dx(1) = car_ob.r*sin_*(u(0)+u(1));
        dx(2) = car_ob.rT*(u(0)-u(1));
        /*std::cout << dx << std::endl;*/
        return dx;
    }

    /*Fを計算する関数*/
    Eigen::VectorXd CalcF(const Eigen::VectorXd &x, const Eigen::VectorXd &U){
        Eigen::VectorXd F = Eigen::VectorXd::Zero(len_u*N);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> XB = Forward(x,U);
        Eigen::MatrixXd X = XB.first;
        Eigen::MatrixXd B_all = XB.second;

        Eigen::MatrixXd Lambda = Backward(X, U);

        for(int i=0; i<N; i++){
            F.segment(len_u*i,len_u) = CalcHu(U.segment(len_u*i,len_u),Lambda.col(i),B_all.block(0,len_u*i,len_x,len_u));
        }
        /*std::cout << "CalcF OK" << std::endl;*/
        return F;
    }

    /*xの予測計算*/
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Forward(const Eigen::VectorXd &x, const Eigen::VectorXd &U){
        Eigen::MatrixXd X(len_x, N+1);
        Eigen::MatrixXd B_all(len_x, len_u*N);
        X.col(0) = x;
        for(int i=1; i<=N; i++){
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> dxB = funcB(X.col(i-1),U.segment(len_u*(i-1),len_u));
            Eigen::VectorXd dx = dxB.first;
            X.col(i) = X.col(i-1) + dt*dx;
            B_all.block(0,len_u*(i-1),len_x,len_u) = dxB.second;
        }
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> XB;
        XB.first = X;
        XB.second = B_all;
        /*std::cout << "Forward OK" << std::endl;*/
        return XB;
    }

    /*随伴変数の計算*/
    Eigen::MatrixXd Backward(const Eigen::MatrixXd &X, const Eigen::VectorXd &U){
        Eigen::MatrixXd Lambda(len_x,N);
        Lambda.col(N-1) = S*(X.col(N)-x_ob);
        for(int i=N-2; i>=0; i--){
            Lambda.col(i) = Lambda.col(i+1) + CalcHx(X.col(i+1),U.segment(len_u*(i+1),len_u),Lambda.col(i+1))*dt;
        }
        /*std::cout << "Backward OK" << std::endl;*/
        return Lambda;
    }

    /*dH/du*/
    Eigen::VectorXd CalcHu(const Eigen::VectorXd &u, const Eigen::VectorXd &lambda, const Eigen::MatrixXd &B){
        Eigen::VectorXd dHdu(len_u);
        dHdu = R*u + B.transpose()*lambda;
        dHdu(0) += 0.15*(2*u(0)-umax(0)-umin(0))/((u(0)-umin(0))*(umax(0)-u(0)));
        dHdu(1) += 0.15*(2*u(1)-umax(1)-umin(1))/((u(1)-umin(1))*(umax(1)-u(1)));
        /*std::cout << "CalcHu OK" << std::endl;*/
        return dHdu;
    }

    /*dH/dx*/
    Eigen::VectorXd CalcHx(const Eigen::VectorXd &x, const Eigen::VectorXd &u, const Eigen::VectorXd &lambda){
        Eigen::MatrixXd dfdx(len_x,len_x);
        Eigen::VectorXd dHdx(len_x);
        dfdx = Calcfx(x,u);
        dHdx = Q*(x-x_ob) + dfdx.transpose()*lambda;
        /*std::cout << "CalcHx OK" << std::endl;*/
        return dHdx;
    }

    /*df/dx*/
    Eigen::MatrixXd Calcfx(const Eigen::VectorXd &x, const Eigen::VectorXd &u){
        Eigen::MatrixXd dfdx(len_x,len_x);

        dfdx << 0.0, 0.0, -car_ob.r*sin(x(2))*(u(0)+u(1)),
        0.0, 0.0, car_ob.r*cos(x(2))*(u(0)+u(1)),
        0.0, 0.0, 0.0;
        
        /*std::cout << "Calcfx OK" << std::endl;*/
        return dfdx;
    }

    /*Givens回転*/
    /*std::pair<Eigen::MatrixXd, Eigen::VectorXd>*/
    void ToUTMat(Eigen::MatrixXd &H, Eigen::VectorXd &gm, const int &m){
        for(int i=0; i<m; i++){
            double nu = sqrt(std::pow(H(i,i),2.0)+std::pow(H(i+1,i),2.0));
            double c_i = H(i,i)/nu;
            double s_i = H(i+1,i)/nu;
            Eigen::MatrixXd Omega = Eigen::MatrixXd::Identity(m+1,m+1);
            Omega(i,i) = c_i;
            Omega(i,i+1) = s_i;
            Omega(i+1,i) = -s_i;
            Omega(i+1,i+1) = c_i;

            H =  Omega*H;
            gm = Omega*gm;
        }

        /*std::pair<Eigen::MatrixXd, Eigen::VectorXd> UTgm;
        UTgm.first = H;
        UTgm.second = gm;*/
        /*std::cout << "ToUTMat OK" << std::endl;*/
        /*return UTgm;*/
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> funcB(const Eigen::VectorXd &x, const Eigen::VectorXd &u){
        Eigen::VectorXd dx(len_x);
        Eigen::MatrixXd B(len_x,len_u);
        double cos_ = cos(x(2));
        double sin_ = sin(x(2));

        B << car_ob.r*cos_, car_ob.r*cos_,
        car_ob.r*sin_, car_ob.r*sin_,
        car_ob.rT, -car_ob.rT;
        
        dx = B*u;
        
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> dxB;
        dxB.first = dx;
        dxB.second = B;
        /*std::cout << "funcB OK" << std::endl;*/
        return dxB;
    }
};



int main(){
    Eigen::VectorXd x(3);

    Eigen::VectorXd x_ob(3);
    x_ob << 3.0, 2.0, 0.0;

    car nonholo_car;
    controller CGMRES_cont(nonholo_car, x_ob);

    double time = 0;

    clock_t start = clock();
    while(time<=20){
        /*画面に表示*/
        /*std::cout << "-------------Inputs-------------\n" << CGMRES_cont.u << std::endl;*/
        std::cout << "-------------Position-------------\n" << CGMRES_cont.x << std::endl;

        /*次の位置を計算する（まだ代入しない）*/
        x = CGMRES_cont.x + nonholo_car.func(CGMRES_cont.u,CGMRES_cont.x)*CGMRES_cont.Ts;
        
        /*時計合わせ*/
        CGMRES_cont.Time = time + CGMRES_cont.Ts;
        
        /*制御入力を更新*/
        CGMRES_cont.CGMRES_control();
        
        /*時刻を進める*/
        time += CGMRES_cont.Ts;
        
        /*座標を更新*/
        CGMRES_cont.x = x;
    }
    clock_t end = clock();
    double loop_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("計算時間：%.2f [s]\n",loop_time);
    
    return 0;
}