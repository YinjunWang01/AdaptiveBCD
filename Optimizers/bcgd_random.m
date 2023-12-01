function [errors, grads] = bcgd_random(X, y, w_init, stepsize, tol, max_iters, block_num, loss, grad_loss, loss_star)
w = w_init;
d = size(w,1);
errors = [];
grads = [];
block_size = d/block_num;

for i = 1:max_iters
    % Compute the gradient
    grad = grad_loss(X, y, w);
    % Sample from uniform distribution
    k = randi(block_num);
    grad_k = zeros(d,1);
    grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
    
    % Update the model parameters
    w = w - stepsize * grad_k;

    errors = [errors (loss(X, y, w) - loss_star)];
    grads = [grads norm(grad)];
    
    % Check the stopping criterion
    if norm(grad) < tol
        break;
    end
end
end