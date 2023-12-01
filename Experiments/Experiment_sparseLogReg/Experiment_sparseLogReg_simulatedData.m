clear;
clc;

rng(1)

% Generate some synthetic data
n = 500;
d = 500;
X = randn(n, d);
w_true = randn(d, 1);
y = sign(X * w_true);
lambda = .05/n;

% Compute L
L = norm(X,2)^2 / (4*n);

% Set up the gradient descent algorithm
w_init = zeros(d, 1);  % initial model parameters
stepsize_fixed = .1/L;  % step size
max_iters = 10000;  % maximum number of iterations
tol = 1e-6;  % stopping criterion tolerance
block_num = 5;
block_update_times = 2;

%% Proximal Gradient -- Compute the f^* + g^*
w_apg_old = w_init;
grad_apg_old = logReg_grad(X, y, w_init);
w_apg = w_init - grad_apg_old;
stepsize_apg_old = 1/L;
theta = 1e9;

for i = 1:max_iters
    grad_apg = logReg_grad(X, y, w_apg);
    norm_w = norm(w_apg - w_apg_old);
    norm_grad = norm(grad_apg - grad_apg_old);
    stepsize_apg = min((sqrt(1 + theta) * stepsize_apg_old), 0.5 * (norm_w / norm_grad));
    theta = stepsize_apg / stepsize_apg_old;

    w_apg_old = w_apg;
    w_apg = w_apg - stepsize_apg * grad_apg;
    w_apg = prox_l1(w_apg, lambda*stepsize_apg);
    stepsize_apg_old = stepsize_apg;
    grad_apg_old = grad_apg;

    if norm(grad_apg) < tol
        break;
    end
end
loss_star = logReg_loss(X, y, w_apg) + lambda * norm(w_apg, 1);

[errors_abcpgn_random2, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
[errors_abcpgn_random3, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
[errors_abcpgn_random4, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
[errors_abcpgn_random5, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
[errors_abcpgn_random6, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
[errors_abcpgn_random7, ~] = abcpgn_random(X, y, w_init, .1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);

semilogy(1:size(errors_abcpgn_random2,2), errors_abcpgn_random2, ...
    1:size(errors_abcpgn_random3,2), errors_abcpgn_random3, ...
    1:size(errors_abcpgn_random4,2), errors_abcpgn_random4, ...
    1:size(errors_abcpgn_random5,2), errors_abcpgn_random5, ...
    1:size(errors_abcpgn_random6,2), errors_abcpgn_random6, ...
    1:size(errors_abcpgn_random7,2), errors_abcpgn_random7, ...
    'LineWidth', 10);
xlabel('Iteration','Interpreter','latex')
ylabel('$f(x_k) - f^{*}$','Interpreter','latex')
%legend({'A-BCPG-n $(N=2)$','A-BCPG-n $(N=3)$','A-BCPG-n $(N=4)$','A-BCPG-n $(N=5)$','A-BCPG-n $(N=6)$','A-BCPG-n $(N=7)$'},...
%    'Location','northeast','Interpreter','latex','FontSize',72)
ax = gca; 
ax.XAxis.FontSize = 48; 
ax.YAxis.FontSize = 48; 
ax.XLabel.FontSize = 84;
ax.YLabel.FontSize = 84;

%{
errors_bcpg_random = bcpg_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_bcpg_cyclic = bcpg_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_bcpg_shuffled = bcpg_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
% Nesterov's Accelerated BCGD
errors_nesterov_random = bcpg_nesterov_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_nesterov_cyclic = bcpg_nesterov_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_nesterov_shuffled = bcpg_nesterov_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
% A-BCGD
errors_abcpg = abcpg_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
% A-BCGD-n
errors_abcpgn = abcpgn_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);


%% Plot
semilogy(1:size(errors_bcpg_random,2), errors_bcpg_random, 1:size(errors_bcpg_cyclic,2), errors_bcpg_cyclic, 1:size(errors_bcpg_shuffled,2), errors_bcpg_shuffled,...
    1:size(errors_nesterov_random,2), errors_nesterov_random, 1:size(errors_nesterov_cyclic,2), errors_nesterov_cyclic, 1:size(errors_nesterov_shuffled,2), errors_nesterov_shuffled,...
    1:size(errors_abcpg,2), errors_abcpg,...
    1:size(errors_abcpgn,2), errors_abcpgn, 'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k)-f^{*}$','Interpreter','latex','FontSize',84)
legend({'BCPG (random)', 'BCPG (cyclic)','BCPG (shuffled)','Nesterov (random)', 'Nesterov (cyclic)', 'Nesterov (shuffled)','A-BCPG','A-BCPG-n (N=2)'},'Location','northeast','Interpreter','latex','FontSize',64)
%}

%{
errors_abcpgn_random = abcpgn_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
errors_abcpgn_cyclic = abcpgn_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
errors_abcpgn_shuffled = abcpgn_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times, lambda);
errors_bcpg_linesearch_random = bcpg_linesearch_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_bcpg_linesearch_cyclic = bcpg_linesearch_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);
errors_bcpg_linesearch_shuffled = bcpg_linesearch_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, lambda);


errors_bcpg_linesearch_random = errors_bcpg_linesearch_random(1,1:max_iters);
%errors_bcpg_linesearch_cyclic = errors_bcpg_linesearch_cyclic(1,1:max_iters);
%errors_bcpg_linesearch_shuffled = errors_bcpg_linesearch_shuffled(1,1:max_iters);


%% Plot

semilogy(1:size(errors_abcpgn_random,2), errors_abcpgn_random, 1:size(errors_abcpgn_cyclic,2), errors_abcpgn_cyclic, 1:size(errors_abcpgn_shuffled,2), errors_abcpgn_shuffled,...
    1:size(errors_bcpg_linesearch_random,2), errors_bcpg_linesearch_random, 1:size(errors_bcpg_linesearch_cyclic,2), errors_bcpg_linesearch_cyclic, 1:size(errors_bcpg_linesearch_shuffled,2), errors_bcpg_linesearch_shuffled, 'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k) - f^{*}$','Interpreter','latex','FontSize',84)
legend({'A-BCPG-n (random, N=2)','A-BCPG-n (cyclic, N=2)','A-BCPG-n (shuffled, N=2)','Linesearch (random)','Linesearch (cyclic)','Linesearch (shuffled)'},'Location','northeast','Interpreter','latex','FontSize',64)
%}




%% Helper Function
function grad = logReg_grad(X, y, w)
% Computes the gradient of logistic regression.
%
% Inputs:
%   X: n x d matrix of training data
%   y: n x 1 vector of training labels (+1 or -1)
%   w: d x 1 vector of model parameters
%
% Output:
%   grad: d x 1 vector of gradient of the logistic regression objective

n = size(X, 1);  % number of training examples
d = size(X, 2);  % number of features

% Compute the gradient
grad = zeros(d, 1);
for i = 1:n
    x_i = X(i, :)';
    y_i = y(i);
    p_i = exp(-y_i * x_i' * w) / (1 + exp(-y_i * x_i' * w));
    grad = grad + (-y_i * p_i * x_i);
end
grad = grad / n;
end

function loss = logReg_loss(X, y, w)
    % Compute the logistic regression loss
    z = X * w;
    loss = mean(log(1 + exp(-y .* z)));
end

function y = prox_l1(x, lambda)
    y = sign(x) .* max(abs(x) - lambda, 0);
end








