function test_example_CNN
%load mnist_uint8;
addpath(genpath('D:/SOFTWARES/Matlab/toolbox/DeepLearnToolbox-master'))
load crack_uint8;

ker=28;
[tm,tn]=size(train_x);
[sm,sn]=size(test_x);
train_x = double(reshape(train_x',ker,ker,tm))/255;
test_x = double(reshape(test_x',ker,ker,sm))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 50;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');

%%
save('crackCnn.mat');
% load('crackCnn.mat');
% A=sim(net,P)%对网络进行仿真
