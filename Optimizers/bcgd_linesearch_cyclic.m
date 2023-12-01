function [errors, grads] = bcgd_linesearch_cyclic(X, y, w_init, stepsize_init, tol, max_iters, block_num, loss, grad_loss, loss_star)
w = w_init;
d = size(w,1);
errors = [];
grads = [];
k = 0;
block_size = d/block_num;

for i = 1:max_iters
    grad = grad_loss(X, y, w);

    k = mod(k,block_num) + 1;
    grad_k = zeros(d,1);
    grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
    
    error_tmp = loss(X, y, w) - loss_star;

    stepsize = 10 * stepsize_init;
    f_cur = loss(X, y, w);
    g_norm2 = norm(grad_k)^2;

    while loss(X, y, w - stepsize * grad_k) > f_cur - 0.0001 * stepsize * g_norm2
        errors = [errors error_tmp];
        stepsize = 0.8 * stepsize;
    end

    w = w - stepsize * grad_k;

    errors = [errors (loss(X, y, w) - loss_star)];
    grads = [grads norm(grad)];
    %errors = [errors loss(X, y, w)];
    
    if norm(grad) < tol
        break;
    end
end
end