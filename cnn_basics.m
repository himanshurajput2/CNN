function [ ] = cnn_basics(  )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

img1  = imread('test4.jpg');
img1_g = rgb2gray(img1);
img_s = im2single(img1_g);
figure(1) ; clf ; imagesc(img_s);

G(: , : , : , 1) = [-1 , 0,1;-2,0,2;-1,0,1];
G(: , : , : , 2) = [-1,-2,-1;0,0,0;1,2,1];
G= single(G) ;


% disp(G);
% size(img_s)
 size(G)
y = vl_nnconv(img_s, G, []) ;
figure(2) ; clf ; vl_imarraysc(y) ; colormap gray ;

% %%%% Different Stride 
y_ds = vl_nnconv(img_s, G, [], 'stride', 8) ;
figure(3) ; clf ; vl_imarraysc(y_ds) ; colormap gray ;

% % %%%% Different Pad
 y_pad = vl_nnconv(img_s, G, [], 'pad', 4) ;
 figure(4) ; clf ; vl_imarraysc(y_pad) ; colormap gray ;

% % %%%% Relu applied on results from last output
z = vl_nnrelu(y) ;
figure(5) ; clf ; vl_imarraysc(z) ; colormap gray ;

z_ds = vl_nnrelu(y_ds) ;
figure(6) ; clf ; vl_imarraysc(z_ds) ; colormap gray ;

z_pad = vl_nnrelu(y_pad) ;
figure(7) ; clf ; vl_imarraysc(z_pad) ; colormap gray ;

% %%% Pool applied on results of last output

p = vl_nnpool(z, 15) ;
figure(8) ; clf ; vl_imarraysc(p) ; colormap gray ;
p_ds = vl_nnpool(z_ds, 15) ;
figure(9) ; clf ; vl_imarraysc(p_ds) ; colormap gray ;
p_pad = vl_nnpool(z_pad, 15) ;
figure(10) ; clf ; vl_imarraysc(p_pad) ; colormap gray;

%%%%%% Different Stride And Pad Setting
y_ds = vl_nnconv(img_s, G, [], 'stride', 4) ;
z_ds = vl_nnrelu(y_ds) ;
p_ds = vl_nnpool(z_ds, 15) ;
figure(11) ; clf ; vl_imarraysc(p_ds) ; colormap gray ;

y_pad = vl_nnconv(img_s, G, [], 'pad', 2) ;
z_pad = vl_nnrelu(y_pad) ;
p_pad = vl_nnpool(z_pad, 15) ;
figure(12) ; clf ; vl_imarraysc(p_pad) ; colormap gray;

end

