function [A_g, b_g, P_g] = optimize_linear_ds_from_data(Data, att_g, fg_type, ctr_type, varargin)
% att_g: global attractor, e.g. att_g = [0 0]'
% fg_type: global dynamics type
% ctr_type: constraint type
%
% [1] Figueroa, N. and Billard, A. (2018) A Physically-Consistent Bayesian Non-Parametric Mixture
%     Model for Dynamical System Learning. In Proceedings of the 2nd Conference on Robot Learning
%     (CoRL). Accepted.


% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% This epsilon is super crucial for stability
epsilon = 0.1;

% Solve the convex optimization problem with Yalmip
sdp_options = []; Constraints = [];
warning('off','YALMIP:strict') 

% Define Variables

b_var = sdpvar(N, 1);
switch fg_type
    case 0 % 0: Fixed Linear system Axi + b
        A_var = sdpvar(N, N, 'diagonal','real');
        P_var = sdpvar(N, N, 'symmetric', 'real');
        A0 = eye(N);
        b0 = -A0*att_g;
        Constraints = [Constraints A_var(1,1) == A_var(2,2)];
        
    case 1 % 0: Approximated Linear system Axi + b
        P_var = sdpvar(N, N, 'symmetric','real');
        A_var = sdpvar(N, N, 'full','real');
end

% Define Constraints
switch ctr_type
    case 0  % (O1) in eq. (9) in [1]
        sdp_options = sdpsettings('solver','sedumi','verbose', 1);
        Constraints = [Constraints A_var' + A_var <= -epsilon*eye(N,N) b_var == -A_var*att_g ];
        P = eye(N);
        
    case 1  % (O2),(O3) in eq. (9) in [1]
        
        % 'penlab': Nonlinear semidefinite programming solver
        sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);        
        
        % Solve Problem with Convex constraints first to get A's
        fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
        [A0, b0] = optimize_linear_ds_from_data(Data, att_g, fg_type, 0);    
        assign(A_var,A0);
        assign(b_var,b0);
        
        if nargin >= 5  % (O3)
            
            Q_var_g = sdpvar(N, N, 'symmetric','real');
            Q_var_l = sdpvar(N, N, 'symmetric','real');
            assign(Q_var_g,-eye(N));
            assign(Q_var_l,-eye(N));
            Constraints = [Constraints, Q_var_g <= -epsilon*eye(N)];
            Constraints = [Constraints, Q_var_l <= -epsilon*eye(N)];
            
            P_g  = varargin{1};  P_l = varargin{3};  % prior estimates; #QEST: why not use P_var?
                                                     % is it not optimized over?
            constraint_type = varargin{5};
            switch constraint_type
                case 'full'
                    % Full Constraints from WSAQF                                       
                    Constraints = [Constraints, transpose(A_var)*P_g + P_g*A_var == Q_var_g];                    
                    Constraints = [Constraints, transpose(A_var)*transpose(P_l) == Q_var_l];
                                   
                case 'global'
                    % Constraint with global component only from P-QLF
                    Constraints = [Constraints, transpose(A_var)*P_g + P_g*A_var == Q_var];
            end
            Constraints = [Constraints, b_var == -A_var*att_g];
            
        else  % (O2)
            assign(P_var,eye(N)); 
            Constraints = [Constraints, transpose(A_var)*P_var + P_var*A_var <= -epsilon*eye(N)];
            Constraints = [Constraints  A_var < -epsilon];  % #QEST: why needed? Proposition 1 in
                                                            % [1] does not mention that requirement
            Constraints = [Constraints, P_var >= epsilon*eye(N,N)];
            Constraints = [Constraints,  b_var == -A_var*att_g];
        end        
end

% Calculate the approximated velocities with A_c
Xi_d_dot = sdpvar(N,M, 'full');
for m = 1:M
    Xi_d_dot(:,m) = A_var*Xi_ref(:,m) + b_var;
end

% Then calculate the difference between approximated velocities
% and the demonstrated ones for A_c
Xi_dot_error = Xi_d_dot - Xi_ref_dot;

% Defining Objective Function depending on constraints
if ctr_type == 0
    Xi_dot_total_error = sdpvar(1,1); Xi_dot_total_error(1,1) = 0;
    for m = 1:M
        Xi_dot_total_error = Xi_dot_total_error + norm(Xi_dot_error(:, m));  % no sqrt in paper?
    end
    Objective = Xi_dot_total_error;
else  % #QEST: not sure why need to handle ctr_type!=0 differently?
    Aux_var     = sdpvar(N,length(Xi_dot_error));
    Objective   = sum((sum(Aux_var.^2)));
    Constraints = [Constraints, Aux_var == Xi_dot_error];
end

% Solve optimization problem
sol = optimize(Constraints, Objective, sdp_options);
if sol.problem ~= 0
    yalmiperror(sol.problem);
end

% Optimization result
sol.info
check(Constraints)
fprintf('Total error: %2.2f\nComputation Time: %2.2f\n', value(Objective),sol.solvertime);

% Output Variables
A_g = value(A_var);
b_g = value(b_var);

if exist('P_var','var')  % P_var always exists? (see LOC 26)
    P_g = value(P_var);
    if ctr_type == 1
        Q = value(Q_var_g);
        P_g = Q;  % #QEST: why?
    end
else
    P_g = P;
end

end


% These constraints don't seem to do anything
% for i=1:M
%     lyap_local_i = (Xi_ref(:,i) - att_g)'*P_l*(Xi_ref(:,i) - att_l);
%     if lyap_local_i >= 0
%         beta_i = 1;
%     else
%         beta_i = 0;
%     end
%     if beta_i == 1
%         Constraints = [Constraints, 2*lyap_local_i*(Xi_ref(:,i) - att_g)'*transpose(A_var)*P_l*(Xi_ref(:,i) - att_l) < 0];
%     end
% end
