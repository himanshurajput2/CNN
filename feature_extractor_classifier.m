function feature_extractor_classifier()

%% 2.1 Using CNNs as a feature extractor and classifier

%% Setting the parameters
opts.dataDir = ('cifar-10-batches-mat') ;
opts.expDir =   ('cifar-data') ;
opts.imdbPath = ('cifar-data\imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 14 ;
opts.train.continue = true ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;

%%        Prepare data
% Reference:  https://github.com/vlfeat/matconvnet/tree/master/examples
% to convert data to the format required by cnn_train

unpackPath = fullfile(opts.dataDir);
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 2]);

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for i = 1:numel(files)
  fd = load(files{i}) ;
  data{i} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{i} = fd.labels' + 1; % Index from 1
  sets{i} = repmat(file_set(i), size(labels{i}));
end
data = single(cat(4, data{:}));
clNames = load('E:\vision\hw-3\q2\cifar-10-batches-mat\batches.meta.mat');
imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = cat(2, sets{:});
imdb.meta.sets = {'train','test'} ;
imdb.meta.classes = clNames.label_names;
save('E:\vision\hw-3\q2\cifar-data\imdb.mat', '-struct', 'imdb') ;

%% Initialize the network
net = initializeCNN() ;
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;

%% Train the network
[net,info] = cnn_train(net, imdb, @getBatch, opts.train) ;

%% Save the network
net.layers(end) = [] ;
net.imageMean = mean(imdb.images.data,4) ;
save('net\net.mat', '-struct', 'net') ;

%% 2.2 Using CNN features and an SVM classifier %%

%% Load the CNN learned before
net2 = load('net\net.mat') ;

%% Show the filters learned in the first layer
figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net2.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

%% Set the training and testing label matrix and instance matrix
count_train = 0;
count_test = 0;
for i = 1:60000
       im = imdb.images.data(:,:,:,i) ;
       res = vl_simplenn(net2, im);
       % Using feature map from layer 8
       fmap= res(9);  
       [d1,d2,d3] = size(fmap.x);   
       mat_tmp = reshape(fmap.x,d1*d2*d3,1);
       mat_tmp = mat_tmp' ;
       [d1,d2] = size(mat_tmp);  
       % Saving image map to training instance matrix %
       % Images in set 1 are traininig images %
        if(imdb.images.set(1,i) == 1)
                 count_train = count_train+1;
                 training_label_vector(count_train,1) = imdb.images.labels(1,i) ;
                 training_instance_matrix(count_train,1:d2) = mat_tmp(1:d2);  
        else
            % Images in set 2 are testing images %
                 count_test = count_test+1;
                 testing_label_vector(count_test,1) = imdb.images.labels(1,i) ;
                 testing_instance_matrix(count_test,1:d2) = mat_tmp(1:d2);                           
        end
                      
end

%% Training the model using linear kernel%%
model = svmtrain(double(training_label_vector), double(training_instance_matrix),'-t 0' );

%% Prediction on Training Images %%
[predicted_label_train,accuracy_train, decision_values_train] = svmpredict(double(training_label_vector), double(training_instance_matrix), model);

%% Prediction on Test Images %%
[predicted_label_test,accuracy_test, decision_values_test] = svmpredict(double(testing_label_vector), double(testing_instance_matrix), model);
       

accuracy_train
accuracy_test
                        

function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;




