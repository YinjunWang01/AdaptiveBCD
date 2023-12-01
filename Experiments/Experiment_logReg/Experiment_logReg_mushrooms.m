clear;
clc;
rng(1)
to_categorical = @(x) categorical(x);

addpath('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Optimizers/');
data = readtable('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Datasets/mushrooms.txt');

varNames = data.Properties.VariableNames; 
for i = 1:length(varNames)
    if iscategorical(data.(varNames{i})) || iscell(data.(varNames{i}))
        data.(varNames{i}) = cellfun(@(x) str2double(x(1:strfind(x, ':')-1)), data.(varNames{i}));
    end
end
data.Var1 = 2*(data.Var1 - 1.5);
data.Var16 = [];
data{:, 2:end} = (table2array(data(:, 2:end)) - mean(table2array(data(:, 2:end)))) ./ (std(table2array(data(:, 2:end)))+1e-10);


% {
n = size(data,1);
d = size(data,2)-1;
X = table2array(data(:,2:end));
y = table2array(data(:,1));

% {
% Compute L
L = norm(X,2)^2 / (4*n);

% Set up the gradient descent algorithm
w_init = zeros(d, 1);  % initial model parameters
stepsize_fixed = .1/L;  % step size
max_iters = 10000;  % maximum number of iterations
tol = 1e-6;  % stopping criterion tolerance
block_num = 4;
block_update_times = 2;


%% Compute the f^*
w_agd_old = w_init;
grad_agd_old = logReg_grad(X, y, w_init);
w_agd = w_init - grad_agd_old;
stepsize_agd_old = 1/L;
theta = 1e9;

for i = 1:max_iters
    grad_agd = logReg_grad(X, y, w_agd);
    norm_w = norm(w_agd - w_agd_old);
    norm_grad = norm(grad_agd - grad_agd_old);
    stepsize_agd = min((sqrt(1 + theta) * stepsize_agd_old), 0.5 * (norm_w / norm_grad));
    theta = stepsize_agd / stepsize_agd_old;

    w_agd_old = w_agd;
    w_agd = w_agd - stepsize_agd * grad_agd;
    stepsize_agd_old = stepsize_agd;
    grad_agd_old = grad_agd;

    if norm(grad_agd) < tol
        break;
    end
end
loss_star = logReg_loss(X, y, w_agd);

[errors_abcgdn_random, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_abcgdn_cyclic, ~] = abcgdn_cyclic(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_abcgdn_shuffled, ~] = abcgdn_shuffled(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_bcgd_linesearch_shuffled, ~] = bcgd_linesearch_shuffled(X, y, w_init, 500/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
[errors_nesterov_shuffled1, ~] = bcgd_nesterov_shuffled(X, y, w_init, 1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
[errors_nesterov_shuffled2, ~] = bcgd_nesterov_shuffled(X, y, w_init, 2/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
[errors_nesterov_shuffled3, ~] = bcgd_nesterov_shuffled(X, y, w_init, 5/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);


%% Plot
semilogy(1:size(errors_abcgdn_random,2), errors_abcgdn_random, ...
    1:size(errors_abcgdn_cyclic,2), errors_abcgdn_cyclic, ...
    1:size(errors_abcgdn_shuffled,2), errors_abcgdn_shuffled,...
    1:size(errors_bcgd_linesearch_shuffled,2), errors_bcgd_linesearch_shuffled,...
    1:size(errors_nesterov_shuffled1,2), errors_nesterov_shuffled1, ...
    1:size(errors_nesterov_shuffled2,2), errors_nesterov_shuffled2, ...
    1:size(errors_nesterov_shuffled3,2), errors_nesterov_shuffled3, ...
    'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k) - f^{*}$','Interpreter','latex','FontSize',84)
ax = gca; 
ax.XAxis.FontSize = 48; 
ax.YAxis.FontSize = 48; 
ax.XLabel.FontSize = 84;
ax.YLabel.FontSize = 84;

%{
[errors_abcgdn_random, ~] = abcgdn_random(X, y, w_init, 1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_abcgdn_cyclic, ~] = abcgdn_cyclic(X, y, w_init, 1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_abcgdn_shuffled, ~] = abcgdn_shuffled(X, y, w_init, 1/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
[errors_bcgd_linesearch_shuffled, ~] = bcgd_linesearch_shuffled(X, y, w_init, 500/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
[errors_nesterov_shuffled1, ~] = bcgd_nesterov_shuffled(X, y, w_init, 2/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
[errors_nesterov_shuffled2, ~] = bcgd_nesterov_shuffled(X, y, w_init, 5/L, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);


%% Plot
semilogy(1:size(errors_abcgdn_random,2), errors_abcgdn_random, 1:size(errors_abcgdn_cyclic,2), errors_abcgdn_cyclic, 1:size(errors_abcgdn_shuffled,2), errors_abcgdn_shuffled,...
    1:size(errors_bcgd_linesearch_shuffled,2), errors_bcgd_linesearch_shuffled, 1:size(errors_nesterov_shuffled1,2), errors_nesterov_shuffled1, 1:size(errors_nesterov_shuffled2,2), errors_nesterov_shuffled2, 'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k) - f^{*}$','Interpreter','latex','FontSize',84)
%}

%{
errors_abcgdn_random = abcgdn_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
errors_abcgdn_cyclic = abcgdn_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
errors_abcgdn_shuffled = abcgdn_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);
errors_bcgd_linesearch_random = bcgd_linesearch_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_bcgd_linesearch_cyclic = bcgd_linesearch_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_bcgd_linesearch_shuffled = bcgd_linesearch_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);


%% Plot
semilogy(1:size(errors_abcgdn_random,2), errors_abcgdn_random, 1:size(errors_abcgdn_cyclic,2), errors_abcgdn_cyclic, 1:size(errors_abcgdn_shuffled,2), errors_abcgdn_shuffled,...
    1:size(errors_bcgd_linesearch_random,2), errors_bcgd_linesearch_random, 1:size(errors_bcgd_linesearch_cyclic,2), errors_bcgd_linesearch_cyclic, 1:size(errors_bcgd_linesearch_shuffled,2), errors_bcgd_linesearch_shuffled, 'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k) - f^{*}$','Interpreter','latex','FontSize',84)
%legend({'A-BCGD-n (random)','A-BCGD-n (cyclic)','A-BCGD-n (shuffled)','Linesearch (random)','Linesearch (cyclic)','Linesearch (shuffled)'},'Location','northeast','Interpreter','latex','FontSize',72)
%}

%{
% BCGD
errors_bcgd_random = bcgd_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_bcgd_cyclic = bcgd_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_bcgd_shuffled = bcgd_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
% Nesterov's Accelerated BCGD
errors_nesterov_random = bcgd_nesterov_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_nesterov_cyclic = bcgd_nesterov_cyclic(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
errors_nesterov_shuffled = bcgd_nesterov_shuffled(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
% A-BCGD
errors_abcgd = abcgd_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star);
% A-BCGD-n
errors_abcgdn = abcgdn_random(X, y, w_init, stepsize_fixed, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, block_update_times);

semilogy(1:size(errors_bcgd_random,2), errors_bcgd_random, 1:size(errors_bcgd_cyclic,2), errors_bcgd_cyclic, 1:size(errors_bcgd_shuffled,2), errors_bcgd_shuffled,...
    1:size(errors_nesterov_random,2), errors_nesterov_random, 1:size(errors_nesterov_cyclic,2), errors_nesterov_cyclic, 1:size(errors_nesterov_shuffled,2), errors_nesterov_shuffled,...
    1:size(errors_abcgd,2), errors_abcgd,...
    1:size(errors_abcgdn,2), errors_abcgdn, 'LineWidth', 10);
xlabel('Iteration','Interpreter','latex','FontSize',84)
ylabel('$f(x_k)-f^{*}$','Interpreter','latex','FontSize',84)
% legend({'BCGD (random)', 'BCGD (cyclic)','BCGD (shuffled)','Nesterov (random)', 'Nesterov (cyclic)', 'Nesterov (shuffled)','A-BCGD','A-BCGD-n (N=2)'},'Location','northeast','Interpreter','latex','FontSize',64)
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
