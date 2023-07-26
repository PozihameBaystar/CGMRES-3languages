close all; clear all; format compact; beep off;%#ok<CLALL> 

Ts = 0.05;  %サンプリング周期[sec]
t_sim_end = 20;   %シミュレーション時間
tsim = 0:Ts:t_sim_end;   %シミュレーションの基準となる時刻

%コントローラーのハイパーパラメーター
cgmres.ht = Ts;    %前進差分近似の時間幅[sec]
cgmres.zeta = 1/Ts; %操作量の安定化ゲインζ
cgmres.tf = 1.0;      %予測時間の最終長さ[sec]
cgmres.alpha = 0.5;  %予測時間の上昇ゲイン
cgmres.N = 20;  %予測区間の分割数

% 評価関数中の重み
cgmres.Q = 100 * [1 0 0; 0 1 0; 0 0 0];
cgmres.R = 1*eye(2);
cgmres.S = 100 * [1 0 0; 0 1 0; 0 0 0];

%これを推測しながら制御する
car.theta = [1.0; 1.0];

%車のパラメーター
car.R = 0.05; %車輪半径[m]
car.T= 0.2; %車輪と車輪の間の幅(車体の横幅)[m]
car.r = car.R/2; %よく使う値を先に計算
car.rT = car.R/car.T;


% 初期条件
car.x = [0;0;0];
cgmres.u = [0;0];

car.len_x = length(car.x);
cgmres.len_u = length(cgmres.u);

cgmres.U = zeros(cgmres.len_u*cgmres.N,1);    %コントローラに与える初期操作量
cgmres.dU = 0;

%目標地点
cgmres.x_ob = [3,2,0]';

%拘束条件
cgmres.umax = [15,15];
cgmres.umin = [-15,-15];

% シミュレーション(ode45)の設定
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);


%描画範囲
area=[-0.2 3 -0.2 3];

tic;
%シミュレーションループ
for i = 1:length(tsim)

    log.u(:,i) = cgmres.u;
    log.x(:,i) = car.x;
    car.x
    
    %シミュレーション計算（最後だけは計算しない）
    if i ~= length(tsim)
        %[t,xi] = ode45( @(t,xi) two_wheel_car(t,xi,cgmres.u,car.R,car.T,car.theta),...
        %    [tsim(i) tsim(i+1)], car.x, opts);
        xi = car.x + func(car.x,cgmres.u,car.R,car.T)*Ts;
    else
        break
    end

    %予測ホライズンを計算
    Tl = cgmres.tf*(1-exp(-cgmres.alpha*Ts*i));

    %平衡状態で必要な入力を計算するがこの条件では影響が無いので計算しない
    
    %IandIを使った適応C/GMRES法コントローラーを関数として実装
    [U, dU] = IandI_CGMRES_control(car,cgmres,Tl);

    cgmres.u = U(1:2);
    cgmres.U = U;
    cgmres.dU = dU;

    
    car.x_preb = car.x;
    %car.x = xi(end,:)'; xの更新法をPython，C++に合わせて単なるオイラー積分にしたのでこっちは使わない
    car.x = xi;

end
toc;


%グラフ化
f=figure();

subplot(5,1,1)
plot(tsim,log.x(1,:),'LineWidth',1); ylabel('x1'); xlabel('Time[s]');
ylim([0,3])
%yticks(0:1:3)
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,2);
plot(tsim,log.x(2,:),'LineWidth',1); ylabel('x2'); xlabel('Time[s]');
ylim([0,2])
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,3);
plot(tsim,log.x(3,:),'LineWidth',1); ylabel('Φ'); xlabel('Time[s]');
%yticks(0:0.1:0.7)
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,4);
plot(tsim,log.u(1,:),'LineWidth',1); ylabel('u1'); xlabel('Time[s]');
%yticks(0:1:11)
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,5);
plot(tsim,log.u(2,:),'LineWidth',1); ylabel('u2'); xlabel('Time[s]');
%yticks(0:1:10)
grid on;
set(gca, 'FontSize', 9.5)



%C/GMRES法コントローラー
function [U, dU] = IandI_CGMRES_control(car,cgmres,Tl)
    
    dt = Tl/cgmres.N; %制御ホライズン内での1ステップを計算
    car.dx = func(car.x,cgmres.u,car.R,car.T);

    %Fの各種を計算
    Fux = CalcF(cgmres.U+cgmres.ht*cgmres.dU, car.x+cgmres.ht*car.dx, cgmres, car, dt);
    Fx = CalcF(cgmres.U, car.x+cgmres.ht*car.dx, cgmres, car, dt);
    F = CalcF(cgmres.U, car.x, cgmres, car, dt);

    %クリロフ部分空間の最初の基底を求める
    left = (Fux - Fx)/cgmres.ht;
    right = -cgmres.zeta*F - ((Fx-F)/cgmres.ht);
    r0 = right - left;

    %1つ目の基底ベクトル
    Vm(:,1) = r0 ./ norm(r0);

     % Arnoldi process(Hm_を作成する)
    Hm = zeros(3,3);
    
    % GMRESの繰り返し回数(基底ベクトルの数)
    m = cgmres.len_u * cgmres.N; 
    
    %Arnoldi法
    for j=1:m
        Fux = CalcF(cgmres.U + ( Vm(:,j) * cgmres.ht ) , ...
                    car.x + (car.dx * cgmres.ht),cgmres,car,dt);        
        Av = ( ( Fux - Fx ) / cgmres.ht );    %Fxtは関数Fの状態微分
        
        %Line 3
        for k=1:j
            Hm(k,j)=Av'*Vm(:,k);
        end    
        
        %Line 4
        temp_vec = 0;
        for k = 1:j
            temp_vec = temp_vec + Hm(k,j).*Vm(:,k);
        end
        
        v_hat = Av - temp_vec;
        
        %Line 5
        Hm(j+1,j) = norm(v_hat);
        
        Vm(:,j+1) = v_hat ./ Hm(j+1,j);
    end

    [UTMat,Omega] = ToUTMat(Hm);
    
    e = zeros( k + 1, 1 );
    e(1) = 1;   
    gm_ = norm(r0)*e;
    for k=1:length(Omega)
        gm_ = Omega{k}*gm_;
    end

    min_y = zeros(length(UTMat)-1,1);   %解の保存先

    for k=length(UTMat)-1 :-1:1
        min_y(k) = (gm_(k) - UTMat(k,:) * min_y)/UTMat(k,k);
    end

    du_new = cgmres.dU + Vm(:,1:m)*min_y;
    
    dU = du_new;
    U = cgmres.U + dU .* cgmres.ht;

end


%Fの計算
function F = CalcF(U,x,cgmres,car,dt)

    %Uを成形
    U_temp = reshape(U,[cgmres.len_u,cgmres.N]);
    
    %xの予測計算＆Bの値の再利用
    [X, B_all] = Forward(x, U_temp, dt, car, cgmres);
   
    %随伴変数の計算
    Lambda = Backward(X, U_temp, dt, car, cgmres);

    %Fのサイズを定める
    F = zeros(cgmres.len_u*cgmres.N,1);

    %Fの算出
    for i=1:cgmres.N
        F(cgmres.len_u*(i-1)+1:cgmres.len_u*i) = CalcHu(U_temp(:,i), Lambda(:,i), cgmres.R, B_all(:,cgmres.len_u*(i-1)+1:cgmres.len_u*i),cgmres.umax,cgmres.umin);
    end
end


%xの予測計算
function [X, B_all] = Forward(x,U,dt,car,cgmres)
    
    X = zeros(car.len_x,cgmres.N+1); %Xのサイズを定義
    B_all = zeros(car.len_x,cgmres.len_u*(cgmres.N)); %B_allのサイズを定義

    X(:,1) = x; 

    for i = 2:cgmres.N+1
        [dx, B] = funcB(X(:,i-1),U(:,i-1),car.r,car.rT);
        X(:,i) = X(:,i-1) + dt*dx;
        B_all(:,cgmres.len_u*(i-2)+1:cgmres.len_u*(i-1)) = B;
    end
end


%随伴変数の計算
function Lambda = Backward(X, U, dt, car, cgmres)

    Lambda = zeros(car.len_x, cgmres.N);
    Lambda(:,cgmres.N) = cgmres.S*(X(:,cgmres.N+1)-cgmres.x_ob);

    for i = cgmres.N-1:-1:1
        Lambda(:,i) = Lambda(:,i+1) + CalcHx(X(:,i+1),cgmres.x_ob,U(:,i+1),Lambda(:,i+1),cgmres.Q,car.R)*dt;
    end
end


%dH/du(の転置)
function dHdu = CalcHu(u,lambda,R,B,umax,umin)
    dHdu = R*u + B'*lambda...
        + 0.15*[(2*u(1)-umax(1)-umin(1))*((u(1)-umin(1))*(umax(1)-u(1)))^(-1); (2*u(2)-umax(2)-umin(2))*((u(2)-umin(2))*(umax(2)-u(2)))^(-1)];
        %+ 1*[(umax(1)<u(1))-(umin(1)>u(1)); (umax(2)<u(2))-(umin(2)>u(2))];
end


%dH/dx(の転置)
function dHdx = CalcHx(x,x_ob,u,lambda,Q,car_R)
    dfdx = Calcfx(x,u,car_R);
    dHdx = Q*(x-x_ob) + dfdx'*lambda;
end


%df/dx(状態方程式の状態変数微分)
function dfdx = Calcfx(x,u,car_R)
    dfdx = [0 0 -(car_R/2)*sin(x(3))*(u(1)+u(2)); 0 0 (car_R/2)*cos(x(3))*(u(1)+u(2)); 0 0 0];
end


function [UTMat,Omega] = ToUTMat(H)
%UNTITLED HをGivens回転でもって上三角行列に変換する
    m = length(H)-1;  %Givens回転する回数
    
    for i=1:m
        nu = sqrt(H(i,i)^2 + H(i+1,i)^2);
        
        c_i = H(i,i)/nu;
        s_i = H(i+1,i)/nu;
        
        Omega{i} = diag(ones(m+1,1));
        Omega{i}(i:i+1,i:i+1) = [c_i,s_i
                                -s_i,c_i];
                            
        H = Omega{i} * H;        
    end
    
    UTMat = H;
end


%差分駆動型二輪車のモデル(ode用)
function dxi = two_wheel_car(t,xi,u,car_R,T)
    dxi = zeros(3,1); %dxiの型を定義
    r = car_R/2;
    rT = car_R/T;
    cos_ = cos(xi(3));
    sin_ = sin(xi(3));
    dxi(1) = r*cos_*u(1) + r*cos_*u(2);
    dxi(2) = r*sin_*u(1) + r*sin_*u(2);
    dxi(3) = rT*u(1) - rT*u(2);
end


%二輪車の状態方程式
function dx = func(xi,u,car_R,T)
    dx = zeros(3,1); %dxiの型を定義
    r = car_R/2;
    rT = car_R/T;
    cos_ = cos(xi(3));
    sin_ = sin(xi(3));
    dx(1) = r * cos_ * (u(1) + u(2));
    dx(2) = r * sin_ * (u(1) + u(2));
    dx(3) = rT * (u(1) - u(2));
end


%二輪車の状態方程式(CalcFで使う用)
function [dx,B] = funcB(xi,u,car_r,car_rT)
    cos_ = cos(xi(3));
    sin_ = sin(xi(3));
    B = [car_r*cos_ car_r*cos_; car_r*sin_ car_r*sin_; car_rT -car_rT];
    dx = B*u;
end
