function [errors, grads] = bcpg_linesearch_random(X, y, w_init, stepsize_init, tol, max_iters, block_num, loss, grad_loss, loss_star, lambda)
w = w_init;
d = size(w,1);
errors = [];
grads = [];
block_size = d/block_num;
loss = @(X,y,w) loss(X,y,w) + lambda * norm(w, 1);

for i = 1:max_iters
    if size(errors,2) >= max_iters
        break;
    end
    grad = grad_loss(X, y, w);

    k = randi(block_num);
    grad_k = zeros(d,1);
    grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
    
    error_tmp = loss(X, y, w) - loss_star;
    
    stepsize = 1 * stepsize_init;
    f_cur = loss(X, y, w);
    g_norm2 = norm(grad_k)^2;
    
    while loss(X, y, prox_l1(w - stepsize * grad_k, lambda*stepsize)) > f_cur - 0.0001 * stepsize * g_norm2
        if size(errors,2) >= max_iters
            break;
        end
        errors = [errors error_tmp];
        stepsize = 0.8 * stepsize;
    end

    w = prox_l1(w - stepsize * grad_k, lambda*stepsize);

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