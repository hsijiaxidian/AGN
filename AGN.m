clear; close all; clc;
warning off;
n = 30;
r = 2;
kappa_list = [1000];
m = 3*n*r;

T = 10000;
eta = 0.5;
thresh_up = 1e10; thresh_low = 1e-14;
errors_GD = zeros(length(kappa_list), T);
errors_AGN = zeros(length(kappa_list), T);


U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(U_seed, r);
As = cell(m, 1);
for k = 1:m
	As{k} = randn(n, n)/sqrt(m);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
  
    y = zeros(m, 1);
    for k = 1:m
        y(k) = As{k}(:)'*X_star(:);
    end
    %% Spectral initialization
    Y = zeros(n, n);
    for k = 1:m
        Y = Y + y(k)*As{k};
    end
    d = 2*r;
    

    %% GD
   L = randn(n,d)/10;
   R = randn(n,d)/10;

    for t = 1:T
        % update L
        X = L*R';  
        error = norm(X - X_star, 'fro');
        errors_GD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end  
        L_plus = L - eta*Z*R;
        R_plus = R - eta*Z'*L;
        L = L_plus;
        R = R_plus;

    end
    
    
    %% AGN
    L = randn(n,d)/10;
    R = randn(n,d)/10;

    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro');%/norm(X_star, 'fro');
        errors_AGN(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        
        % update L
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        Delta_L = mldivide(R, Z');
        L = L - eta*Delta_L';
        
        % update R
        X = L*R';
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end 
        Delta_R = mldivide(L,Z);
        R = R - eta*Delta_R';

    end
        
  
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 1:1:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{GD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_AGN(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 1:1:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AGN}~\\kappa=%d$', kappa);
end

xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MS_n=%d_r=%d_m=%d', n, r, m);



