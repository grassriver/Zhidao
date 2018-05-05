function u = utility(w, mu, sigma, l)
u = - w'*mu + l*w'*sigma*w;
end