mu = rand(2,1);
sigma = rand(2,2);
sigma = sigma+sigma';
sigma = sigma+max(sigma)*eye(2,2);
fun = @(w)utility(w, mu, sigma, l);

A = [];
b = [];
Aeq = [];
beq = [];
lb = [0;0];
ub = [1;1];
w0 = [0.5; 0.5];
w = fmincon(fun, w0, A, b, Aeq, beq, lb, ub);