clear; close all; clc;
warning off;

% Parameters
n = 30; % Dimension of the matrix
r = 2; % Rank of the ground truth matrix
kappa_list = [1000]; % Condition number(s) for the ground truth matrix
m = 3 * n * r; % Number of measurements

% Optimization settings
T = 10000; % Maximum number of iterations
eta = 0.5; % Learning rate
thresh_up = 1e10; % Upper threshold for error (divergence)
thresh_low = 1e-14; % Lower threshold for error (convergence)

% Initialize error storage
errors_GD = zeros(length(kappa_list), T);
errors_AGN = zeros(length(kappa_list), T);

% Generate ground truth matrix X_star
U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r); % Orthogonalize U_star
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(V_seed, r); % Orthogonalize V_star

% Generate measurement matrices {A_k}
As = cell(m, 1);
for k = 1:m
    As{k} = randn(n, n) / sqrt(m); % Normalized random measurement matrices
end

% Main loop over condition numbers
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r); % Singular values of X_star
    L_star = U_star * diag(sqrt(sigma_star)); % Left factor of X_star
    R_star = V_star * diag(sqrt(sigma_star)); % Right factor of X_star
    X_star = L_star * R_star'; % Ground truth matrix

    % Generate measurements y = A_k(X_star)
    y = zeros(m, 1);
    for k = 1:m
        y(k) = As{k}(:)' * X_star(:);
    end

    % Spectral initialization
    Y = zeros(n, n);
    for k = 1:m
        Y = Y + y(k) * As{k};
    end
    d = 2 * r; % Overparameterization factor

    %% Gradient Descent (GD)
    L = randn(n, d) / 10; % Initialize L
    R = randn(n, d) / 10; % Initialize R

    for t = 1:T
        X = L * R'; % Current estimate
        error = norm(X - X_star, 'fro'); % Frobenius norm error
        errors_GD(i_kappa, t) = error;

        % Check for convergence or divergence
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end

        % Compute gradient update
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)' * X(:) - y(k)) * As{k};
        end
        L_plus = L - eta * Z * R; % Update L
        R_plus = R - eta * Z' * L; % Update R
        L = L_plus;
        R = R_plus;
    end

    %% Approximate Gauss-Newton (AGN)
    L = randn(n, d) / 10; % Reinitialize L
    R = randn(n, d) / 10; % Reinitialize R

    for t = 1:T
        X = L * R'; % Current estimate
        error = norm(X - X_star, 'fro'); % Frobenius norm error
        errors_AGN(i_kappa, t) = error;

        % Check for convergence or divergence
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end

        % Compute AGN update for L
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)' * X(:) - y(k)) * As{k};
        end
        Delta_L = mldivide(R, Z'); % Solve for Delta_L
        L = L - eta * Delta_L'; % Update L

        % Compute AGN update for R
        X = L * R';
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)' * X(:) - y(k)) * As{k};
        end
        Delta_R = mldivide(L, Z); % Solve for Delta_R
        R = R - eta * Delta_R; % Update R
    end
end

%% Plot results
clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]}; % Colors for plotting
mks = {'o', 'x', 'p', 's', 'd'}; % Markers for plotting
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};

% Plot GD errors
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    errors = errors(errors > thresh_low); % Filter out converged/diverged points
    t_subs = 1:1:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{GD}~\\kappa=%d$', kappa);
end

% Plot AGN errors
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_AGN(i_kappa, :);
    errors = errors(errors > thresh_low); % Filter out converged/diverged points
    t_subs = 1:1:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AGN}~\\kappa=%d$', kappa);
end

% Add labels and legend
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);

% Save figure
fig_name = sprintf('MS_n=%d_r=%d_m=%d', n, r, m);
saveas(gcf, fig_name, 'png');
