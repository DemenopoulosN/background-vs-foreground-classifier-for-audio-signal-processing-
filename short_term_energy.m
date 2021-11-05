function E = short_term_energy(x,N,L)

m = 0;
E = [];
while m*L+N-1+1<=length(x)
    E = [E sum(x(m*L+1:m*L+N-1+1).^2)/N];
    m = m+1;
end