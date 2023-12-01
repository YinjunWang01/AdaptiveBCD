function [errors, grads] = abcgd_cyclic(X, y, w_init, stepsize0, tol, max_iters, block_num, loss, grad_loss, loss_star)
w_old = w_init;
grad_old = grad_loss(X, y, w_init);
w = w_init - grad_old;
stepsize_old = stepsize0;
theta = 1e9;
d = size(w,1);
errors = zeros(1, max_iters);
grads = zeros(1, max_iters);
block_size = d/block_num;

for i = 1:(max_iters/block_num)
    for k = 1:block_num
    grad = grad_loss(X, y, w);

    grad_k = zeros(d,1);
    grad_k((1+(k-1)*block_size:block_size+(k-1)*block_size)) = grad((1+(k-1)*block_size:block_size+(k-1)*block_size));

    % Compute stepsize
    norm_w = norm(w - w_old);
    norm_grad = norm(grad - grad_old);
    stepsize = min((sqrt(1 + theta) * stepsize_old), 0.5 * (norm_w / norm_grad));
    theta = stepsize / stepsize_old;

    w_old = w;
    w = w - stepsize * grad_k;
    stepsize_old = stepsize;
    grad_old = grad;

    errors(1,i) = loss(X, y, w) - loss_star;
    grads(1,i) = norm(grad);

    if norm(grad) < tol
        break;
    end
    end
end
errors = errors(1,1:i);
grads = grads(1,1:i);
end