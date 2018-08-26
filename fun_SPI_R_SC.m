function [x,s] = fun_SPI_R_SC(kernels,patterns, y,signal, rho_0,rho_factor,rho_max,maxinter_in,...
    maxiter_out,lambda,verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% kernels: convolutional filters (patch * patch * numbers)
% patterns: illumination patterns (pixels * pixels * pattern numbers)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[size_h,size_w,t]=size(patterns);
mask=reshape(patterns,[size_h*size_w,t]);
mask=mask';
filtersize=size(kernels);
k=filtersize(3);
psfradius=floor([filtersize(1)/2, filtersize(2)/2]);
% y: single pixel measurements (collum vector)

% Outputs:
% im_r: reconstructed image (pixels * pixels)
% size_s=[size_h+2*psfradius(1),size_w+2*psfradius(2),k];
size_s=[size_h,size_w,k];
ProxSparse=@(u,theta)max(0,1-theta./abs(u)).*u;

% initialize
s=rand(size_s); s_hat=rand(size_s);
d=kernels; d_hat=psf2otf(d,size_s);
u=real(ifft2(s_hat.*d_hat));
v=s;

a=rand(size_s);
b=rand(size_s);

rho=rho_0;
iter_out=0;

while iter_out< maxiter_out
    
    
    % optimize each k (sk, uk, ...)
    for i=1:k
        iter_in=0;
        while iter_in<maxinter_in
            % optimize s
            t1=u(:,:,i)+1/rho*a(:,:,i); t1_hat=fft2(t1);
            t2=v(:,:,i)+1/rho*b(:,:,i); t2_hat=fft2(t2);
            s_hat(:,:,i)=  (conj(d_hat(:,:,i)).*t1_hat+t2_hat) ./ (conj(d_hat(:,:,i)).*d_hat(:,:,i)+ones(size_s(1),size_s(2)) );
            s(:,:,i)=real(ifft2(s_hat(:,:,i)));
            
            % optimize u
            
            tmpu=u;
            tmpu(:,:,i)=zeros(size_s(1),size_s(2));
            tmpu_sum=sum(tmpu,3);
            t1=y-mask*tmpu_sum(:);
            t2=real(ifft2(s_hat(:,:,i).*d_hat(:,:,i)))-1/rho*a(:,:,i);
            u_vec=(2*(mask'*mask)+rho*eye(size_s(1)*size_s(2)))\(2*mask'*t1+rho*t2(:));
            u(:,:,i)=reshape(u_vec,[size_s(1),size_s(2)]);
            
            
            % optimize v
            v(:,:,i)=ProxSparse(s(:,:,i)-1/rho*b(:,:,i),lambda/rho);
            
            % update a,b
            tt=real(ifft2(d_hat(:,:,i).*s_hat(:,:,i)));
            a(:,:,i)=a(:,:,i)+rho*(u(:,:,i)-tt);
            b(:,:,i)=b(:,:,i)+rho*(v(:,:,i)-tt);
            
            % update rho
            rho=min(rho_max,rho_factor*rho);
            
            
            iter_in=iter_in+1;
                                  
        end
    end
    
    iter_out=iter_out+1;
  
    if verbose
        xtmp=real(ifft2(d_hat.*s_hat));
        x=sum(xtmp,3);
        res=norm(y-mask*x(:));
        figure(1),
        subplot(121),imshow(signal,[]);
        subplot(122),imshow(x,[]);
        fprintf('%d interation of the whole %d times is finished, the error is %d\n',iter_out,maxiter_out,res);
    end
   
end










end










