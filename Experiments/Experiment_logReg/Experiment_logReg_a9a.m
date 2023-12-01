% {
clear;
clc;
rng(1)
to_categorical = @(x) categorical(x);

addpath('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Optimizers/');
data = readtable('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Datasets/a9a.txt');

% {
varNames = data.Properties.VariableNames; 
for i = 1:length(varNames)
    if iscategorical(data.(varNames{i})) || iscell(data.(varNames{i}))
        data.(varNames{i}) = cellfun(@(x) str2double(x(1:strfind(x, ':')-1)), data.(varNames{i}));
    end
end
data.Var12 = [];
data.Var13 = [];
data.Var14 = [];
data.Var15 = [];
data{:, 2:end} = (table2array(data(:, 2:end)) - mean(table2array(data(:, 2:end)))) ./ (std(table2array(data(:, 2:end)))+1e-10);


% {
n = size(data,1);
d = size(data,2)-1;
X = table2array(data(:,2:end));
y = table2array(data(:,1));
X = X(1:16200,:);
y = y(1:16200,:);

%{
% Compute L
L = norm(X,2)^2 / (4*n);

% Set up the gradient descent algorithm
w_init = zeros(d, 1);  % initial model parameters
stepsize_fixed = 1/L;  % step size
max_iters = 10000;  % maximum number of iterations
tol = 1e-6;  % stopping criterion tolerance
block_num = 5;
block_update_times = 2;
%}

%{
clear;
clc;
rng(1)
to_categorical = @(x) categorical(x);

addpath('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Optimizers/');
data = readtable('/Users/yinjunwang/Documents/MATLAB/AdaptiveBCD/Datasets/ijcnn');
data.Var2 = [];

varNames = data.Properties.VariableNames;  % Get the variable names
for i = 1:length(varNames)
    if ~strcmp(varNames{i}, 'Var1') 
        % Split each string at the colon and convert the second part to a number
        data.(varNames{i}) = cellfun(@(x) str2double(x{2}), cellfun(@(x) strsplit(x, ':'), data.(varNames{i}), 'UniformOutput', false));
    end
end
data{:, 2:end} = (table2array(data(:, 2:end)) - mean(table2array(data(:, 2:end)))) ./ (std(table2array(data(:, 2:end)))+1e-10);


n = size(data,1);
d = size(data,2)-1;
X = table2array(data(:,2:end));
y = table2array(data(:,1));

% Compute L
L = norm(X,2)^2 / (4*n);

% Set up the gradient descent algorithm
w_init = randn(d,1);  % initial model parameters
stepsize_fixed = 1/L;  % step size
max_iters = 10000;  % maximum number of iterations
tol = 1e-6;  % stopping criterion tolerance
block_num = 4;
block_update_times = 2;
%}


%% Compute the f^*
%{
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
%}

%{
[errors_abcgdn_random2, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 2);
[errors_abcgdn_random3, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 3);
[errors_abcgdn_random4, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 4);
[errors_abcgdn_random5, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 5);
[errors_abcgdn_random6, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 6);
[errors_abcgdn_random7, ~] = abcgdn_random(X, y, w_init, 1e-10, tol, max_iters, block_num, @logReg_loss, @logReg_grad, loss_star, 7);

semilogy(1:size(errors_abcgdn_random2,2), errors_abcgdn_random2, ...
    1:size(errors_abcgdn_random3,2), errors_abcgdn_random3, ...
    1:size(errors_abcgdn_random4,2), errors_abcgdn_random4, ...
    1:size(errors_abcgdn_random5,2), errors_abcgdn_random5, ...
    1:size(errors_abcgdn_random6,2), errors_abcgdn_random6, ...
    1:size(errors_abcgdn_random7,2), errors_abcgdn_random7, ...
    'LineWidth', 10);
xlabel('Iteration','Interpreter','latex')
ylabel('$f(x_k) - f^{*}$','Interpreter','latex')
legend({'A-BCGD-n $(N=2)$','A-BCGD-n $(N=3)$','A-BCGD-n $(N=4)$','A-BCGD-n $(N=5)$','A-BCGD-n $(N=6)$','A-BCGD-n $(N=7)$'},...
    'Location','northeast','Interpreter','latex','FontSize',72)
ax = gca; 
ax.XAxis.FontSize = 48; 
ax.YAxis.FontSize = 48; 
ax.XLabel.FontSize = 84;
ax.YLabel.FontSize = 84;
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
%}