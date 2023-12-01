function [errors, grads] = bcpg_nesterov_shuffled(X, y, w_init, stepsize, tol, max_iters, block_num, loss, grad_loss, loss_star, lambda)
w1 = w_init;
w1_old = w1;
w2 = w_init;
d = size(w1,1);
errors = [];
grads = [];
block_size = d/block_num;
loss = @(X,y,w) loss(X,y,w) + lambda * norm(w, 1);

for i = 1:max_iters/block_num
    rand_coords = randperm(block_num);
    for j = 1:block_num
    % Compute the gradient
    grad = grad_loss(X, y, w2);
    % Sample from uniform distribution
    k = rand_coords(j);
    grad_k = zeros(d,1);
    grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
    
    % Update the model parameters
    w1_old = w2;
    w1 = w2 - stepsize * grad_k;
    w1 = prox_l1(w1, lambda*stepsize);
    w2 = w1 + ((i-1)/(i+2)) * (w1 - w1_old);

    errors = [errors (loss(X, y, w2) - loss_star)];
    grads = [grads norm(grad)];
    
    % Check the stopping criterion
    if (loss(X, y, w2) - loss_star) < tol
        break;
    end
    end
end

function y = prox_l1(x, lambda)
    y = sign(x) .* max(abs(x) - lambda, 0);
end

end