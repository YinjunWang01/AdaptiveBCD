function [errors, grads] = abcpg_random(X, y, w_init, stepsize0, tol, max_iters, block_num, loss, grad_loss, loss_star, lambda)
w_old = w_init;
grad_old = grad_loss(X, y, w_init);
w = w_init - grad_old;
stepsize_old = stepsize0;
theta = 1e9;
d = size(w,1);
errors = [];
grads = [];
block_size = d/block_num;
loss = @(X,y,w) loss(X,y,w) + lambda * norm(w, 1);

for i = 1:max_iters
    grad = grad_loss(X, y, w);

    % Sample from uniform distribution
    k = randi(block_num);
    grad_k = zeros(d,1);
    grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);

    % Compute stepsize
    norm_w = norm(w - w_old);
    norm_grad = norm(grad - grad_old);
    stepsize = min((sqrt(1 + theta) * stepsize_old), 0.5 * (norm_w / norm_grad));
    theta = stepsize / stepsize_old;

    w_old = w;
    w = w - stepsize * grad_k;
    w = prox_l1(w, lambda*stepsize);
    stepsize_old = stepsize;
    grad_old = grad;

    errors = [errors (loss(X, y, w) - loss_star)];
    grads = [grads norm(grad)];

    if (loss(X, y, w) - loss_star) < tol
        break;
    end
end

function y = prox_l1(x, lambda)
    y = sign(x) .* max(abs(x) - lambda, 0);
end

end

