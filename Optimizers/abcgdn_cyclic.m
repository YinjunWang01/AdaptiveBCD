function [errors, grads] = abcgdn_cyclic(X, y, w_init, stepsize0, tol, max_iters, block_num, loss, grad_loss, loss_star, block_update_times)
w_old = w_init;
grad_old = grad_loss(X, y, w_init);
w = w_init - stepsize0 * grad_old;
grad = grad_loss(X, y, w);
d = size(w,1);
block_size = d/block_num;
theta = ones(block_num,1) * 1e9;
% saved for computing local smoothness when n = 1
w_pre = w_old;       
grad_pre = grad_old; 
% 
errors = zeros(1, max_iters);
grads = zeros(1, max_iters);

% initialize stepsizes
stepsizes_old = zeros(block_num,1);
for k = 1:block_num
    w_tmp = w((1+(k-1)*block_size:block_size+(k-1)*block_size));
    w_old_tmp = w_old((1+(k-1)*block_size:block_size+(k-1)*block_size));
    grad_tmp = grad((1+(k-1)*block_size:block_size+(k-1)*block_size));
    grad_old_tmp = grad_old((1+(k-1)*block_size:block_size+(k-1)*block_size));

    norm_w = norm(w_tmp - w_old_tmp);
    norm_grad = norm(grad_tmp - grad_old_tmp);
    % stepsizes_old(k,1) = (1/2) * (norm_w / norm_grad); 
    stepsizes_old(k,1) = (1/sqrt(2)) * (norm_w / norm_grad);
end

%% main iteration
for i = 1:(max_iters/block_update_times)
    k = mod(k,block_num)+1;
    for j = 1:block_update_times
        iter = block_update_times * (i - 1) + j;
        if j == 1 
            grad_k = zeros(d,1);
            grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
            %
            w_tmp = w((1+(k-1)*block_size:block_size+(k-1)*block_size));
            w_pre_tmp = w_pre((1+(k-1)*block_size:block_size+(k-1)*block_size));
            grad_tmp = grad((1+(k-1)*block_size:block_size+(k-1)*block_size));
            grad_pre_tmp = grad_pre((1+(k-1)*block_size:block_size+(k-1)*block_size));

            norm_w = norm(w_tmp - w_pre_tmp);
            norm_grad = norm(grad_tmp - grad_pre_tmp);
            
            % stepsize = min((sqrt(1 + theta(k,1)) * stepsizes_old(k,1)), (1/2) * (norm_w / norm_grad));
            stepsize = min((sqrt(1 + theta(k,1)) * stepsizes_old(k,1)), (1/sqrt(2)) * (norm_w / norm_grad));
            theta(k,1) = stepsize / stepsizes_old(k,1);

            w_old = w;
            w = w - stepsize * grad_k;
            grad_old = grad;
            grad = grad_loss(X, y, w);
            stepsizes_old(k,1) = stepsize;

            errors(1,iter) = loss(X, y, w) - loss_star;
            grads(1,iter) = norm(grad);
            
            if norm(grad) < tol
                break;
            end
        else
            grad_k = zeros(d,1);
            grad_k(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad(1+(k-1)*block_size:block_size+(k-1)*block_size);
            % 
            w_tmp = w((1+(k-1)*block_size:block_size+(k-1)*block_size));
            w_old_tmp = w_old((1+(k-1)*block_size:block_size+(k-1)*block_size));
            grad_tmp = grad((1+(k-1)*block_size:block_size+(k-1)*block_size));
            grad_old_tmp = grad_old((1+(k-1)*block_size:block_size+(k-1)*block_size));

            norm_w = norm(w_tmp - w_old_tmp);
            norm_grad = norm(grad_tmp - grad_old_tmp);
            % stepsize = min((sqrt(1 + theta(k,1)) * stepsizes_old(k,1)), (1/2) * (norm_w / norm_grad));
            stepsize = min((sqrt(1 + theta(k,1)) * stepsizes_old(k,1)), (1/sqrt(2)) * (norm_w / norm_grad));
            theta(k,1) = stepsize / stepsizes_old(k,1);

            w_old = w;
            w = w - stepsize * grad_k;
            grad_old = grad;
            grad = grad_loss(X, y, w);
            stepsizes_old(k,1) = stepsize;

            if j == block_update_times
                w_pre(1+(k-1)*block_size:block_size+(k-1)*block_size) = w_old(1+(k-1)*block_size:block_size+(k-1)*block_size);
                grad_pre(1+(k-1)*block_size:block_size+(k-1)*block_size) = grad_old(1+(k-1)*block_size:block_size+(k-1)*block_size);
            end

            errors(1,iter) = loss(X, y, w) - loss_star;
            grads(1,iter) = norm(grad);
  
            if norm(grad) < tol
                break;
            end
        end
    end
    if norm(grad) < tol
        break;
    end
end
iter = block_update_times * (i - 1) + j;
errors = errors(1,1:iter);
grads = grads(1,1:iter);
end


