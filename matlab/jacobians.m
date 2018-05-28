%% Symbols 
syms dq0 dq1 dq2 dq3 dq4 dq5 dq6 dq7 dq8 dq9 real
syms psii q_1 q_2 q_3 real
syms dx dpsi dq_1 dq_2 dq_3 real
syms R L L1 L2 L3 real

%% Time Differentiation
syms t dPSI dQ_1 dQ_2 dQ_3 real
syms PSI(t) Q_1(t) Q_2(t) Q_3(t)
dPSI=diff(PSI,t); dQ_1=diff(Q_1,t); dQ_2=diff(Q_2,t); dQ_3=diff(Q_3,t); 

mydiff = @(H) formula(subs(diff(symfun(subs(H,...
    [psii,q_1,q_2,q_3],...
    [PSI,Q_1,Q_2,Q_3]),t),t),...
    [PSI,Q_1,Q_2,Q_3,dPSI,dQ_1,dQ_2,dQ_3],...
    [psii,q_1,q_2,q_3,dpsi,dq_1,dq_2,dq_3]));

%% Full Jacobian using DART's coordinates
w1.frame1 = [dq0 dq1 dq2]';
v1.frame1 = [dq3 dq4 dq5]';
w2_local.frame2 = [-dq8 0 0]';
w3_local.frame3 = [0 0 dq9]';

rot1.world = [sin(psii), sin(q_1)*cos(psii), -cos(q_1)*cos(psii); ...
             -cos(psii), sin(q_1)*sin(psii), -cos(q_1)*sin(psii); ...
                      0,           cos(q_1),            sin(q_1)];
rot2.frame1 = [[1 0 0]', [0 cos(q_2) -sin(q_2)]', [0 cos(q_2) sin(q_2)]'];
rot3.frame2 = [[cos(q_3) sin(q_3) 0]', [-sin(q_3) cos(q_3) 0]', [0 0 1]'];

P2.frame1 = [0 L1 0]';
P3.frame2 = [0 L2 0]';
PEE.frame3 = [0 L3 0]';
PEE.frame2 = P3.frame2 + rot3.frame2*PEE.frame3;
PEE.frame1 = P2.frame1 + rot2.frame1*PEE.frame2;

vEE.world = rot1.world*(v1.frame1 + cross(w1.frame1, PEE.frame1) ...
                       + rot2.frame1*(cross(w2_local.frame2, PEE.frame2) ...
                                      + rot3.frame2*cross(w3_local.frame3, PEE.frame3)));
JEE_dart = sym(zeros(3, 10));
dqVec = [dq0 dq1 dq2 dq3 dq4 dq5 dq6 dq7 dq8 dq9];
for i=1:3
    for j=1:10
        JEE_dart(i, j) = diff(vEE.world(i), dqVec(j));
    end
end

%% Transform to full Jacobian using minimum coordinates
% Coordinate Transformation to minimum set of coordinates
% dq0 = -dq_1
% dq1 = dpsi*cos(q_1)
% dq2 = dpsi*sin(q_1)
% dq3 = 0
% dq4 = dx*sin(q_1)
% dq5 = -dx*cos(q_1)
% dq6 = dx/R - (L/(2*R))*dpsi - dq_1
% dq7 = dx/R + (L/(2*R))*dpsi - dq_1
% dq8 = dq_2
% dq9 = dq_3
% [dq0 dq1 dq2 dq3 dq4 dq5 dq6 dq7]' = J*[dx dpsi dq_1]'; 
% where
J = [        0,        0,        -1, 0, 0;
             0, cos(q_1),         0, 0, 0;
             0, sin(q_1),         0, 0, 0;
             0,        0,         0, 0, 0;
      sin(q_1),        0,         0, 0, 0;
     -cos(q_1),        0,         0, 0, 0;
           1/R, -L/(2*R),        -1, 0, 0;
           1/R,  L/(2*R),        -1, 0, 0;
             0,        0,         0, 1, 0;
             0,        0,         0, 0, 1];

dJ = [       0,         0,         0, 0, 0;
             0, -sin(q_1),         0, 0, 0;
             0,  cos(q_1),         0, 0, 0;
             0,         0,         0, 0, 0;
      cos(q_1),         0,         0, 0, 0;
      sin(q_1),         0,         0, 0, 0;
             0,         0,         0, 0, 0;
             0,         0,         0, 0, 0;
             0,        0,          0, 0, 0;
             0,        0,          0, 0, 0]*(dq_1);

JEE_min_transform = simplify(JEE_dart*J);

%% Derive Jacobian in minimum set of coordinates using first principles
w0.frame0 = [0 0 dpsi]';
v0.frame0 = [dx 0 0]';
w1_local.frame1 = [-dq_1 0 0]';
w2_local.frame2 = [-dq_2 0 0]';
w3_local.frame3 = [0 0 dq_3]';

rot0.world = [[cos(psii) sin(psii) 0]', [-sin(psii) cos(psii) 0]', [0 0 1]'];
rot1.frame0 = [[0 -1 0]', [sin(q_1) 0 cos(q_1)]', [-cos(q_1) 0 sin(q_1)]'];
rot2.frame1 = [[1 0 0]', [0 cos(q_2) -sin(q_2)]', [0 cos(q_2) sin(q_2)]'];
rot3.frame2 = [[cos(q_3) sin(q_3) 0]', [-sin(q_3) cos(q_3) 0]', [0 0 1]'];

P2.frame1 = [0 L1 0]';
P3.frame2 = [0 L2 0]';
PEE.frame3 = [0 L3 0]';
PEE.frame2 = P3.frame2 + rot3.frame2*PEE.frame3;
PEE.frame1 = P2.frame1 + rot2.frame1*PEE.frame2;
PEE.frame0 = rot1.frame0*PEE.frame1;

% vEE = v0 + cross(w0 + w1_local, P2) + cross(w0 + w1_local + w2_local, P3) ...
%     + cross(w0 + w1_local + w2_local + w3_local, PEE);
vEE.world = rot0.world*(v0.frame0 + cross(w0.frame0, PEE.frame0) ...
    + rot1.frame0*(cross(w1_local.frame1, PEE.frame1) ...
    + rot2.frame1*(cross(w2_local.frame2, PEE.frame2) ...
    + rot3.frame2*cross(w3_local.frame3, PEE.frame3))));

JEE_min = sym(zeros(3, 5));
dqVec = [dx dpsi dq_1 dq_2 dq_3];
for i=1:3
    for j=1:5
        JEE_min(i, j) = diff(vEE.world(i), dqVec(j));
    end
end

%% Jacobian we want
vEE_frame0 = rot1.frame0*(cross(w1_local.frame1, PEE.frame1) ...
    + rot2.frame1*(cross(w2_local.frame2, PEE.frame2) ...
    + rot3.frame2*cross(w3_local.frame3, PEE.frame3)));

JEE_body = sym(zeros(3, 3));
dqVec = [dq_1 dq_2 dq_3];
for i=1:3
    for j=1:3
        JEE_body(i, j) = diff(vEE_frame0(i), dqVec(j));
    end
end

%% Deducing the Jacobian we want from JEE_min
JEE_body_deduced = rot0.world'*JEE_min(:,3:5);

%% Derivative of Jacobian we want
dJEE_body = simplify(mydiff(JEE_body));
dJEE_body_comps = sym(zeros(3,3,4));
dqVec = [dpsi dq_1 dq_2 dq_3]';    

%% Deducing the derivative of Jacobian we want
drot0.world = mydiff(rot0.world);
dJEE_dart = mydiff(JEE_dart);
dJEE_min_frame0 = simplify(drot0.world'*JEE_dart*J + rot0.world'*dJEE_dart*J + rot0.world'*JEE_dart*dJ);
dJEE_body_deduced = dJEE_min_frame0(:,3:5);

%% Deducing task speed
vEE_frame0_deduced = rot0.world'*(vEE.world) - v0.frame0 + drot0.world'*(rot0.world*PEE.frame0);