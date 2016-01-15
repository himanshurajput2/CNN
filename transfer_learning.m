
net = load('imagenet-caffe-alex.mat');
%% newNet in the required format
newNet = net;
for i = 1: numel(net.layers)
    if strcmp(net.layers{1,i}.type,'conv')
        newNet.layers{1,i} = rmfield(newNet.layers{1,i},'weights');
        newNet.layers{1,i}.filters = net.layers{1,i}.weights{1,1};
        newNet.layers{1,i}.biases = net.layers{1,i}.weights{1,2};
    end
end

%% Parsing all files and dividing into training and testing set
% Also setting training and testing matrix at same time
    caltechPath = 'cropped';
    category = dir(caltechPath);
    count_train = 0;
    count_test = 0;
    for i = 1:length(category)
        if category(i).name(1) == '.' 
            continue
        else
            fprintf('processing %s\n', category(i).name);
            imageFile = dir(fullfile(caltechPath, category(i).name,'image*'));
            len = length(imageFile);
            % If number of image if more than 50 use only 50 images %
            if( len > 50)
                len = 50;
            end
            for j = 1:len
                if imageFile(j).name(1) == '.'
                    continue
                else                  
                    im = imread(fullfile(caltechPath,category(i).name, imageFile(j).name));
                    im = im2single(im) ;
                    res = vl_simplenn(newNet, im);
                    % Get feature map for 19th layer %
                    fmap= res(20);      
                    %  Use 3/5 images for training and rest for testing %
                   if(j<=3/5 * len)   
                          count_train = count_train+1;
                          %  Assigning the label for the image %
                          %  Label is same for all images of a class %
                          training_label_vector(count_train,1) = i-2;                      
                          [d1,d2,d3] = size(fmap.x);   
                          mat_tmp = reshape(fmap.x,d1*d2*d3,1);
                          mat_tmp = mat_tmp' ;
                          [d1,d2] = size(mat_tmp);  
                          % Saving image map to training instance matrix %
                          training_instance_matrix(count_train,1:d2) = mat_tmp(1:d2);                    
                   else
                        %  Use 2/5 images for testing %
                          count_test = count_test+1;
                          testing_label_vector(count_test,1) = i-2;
                          [d1,d2,d3] = size(fmap.x);    
                          mat_tmp = reshape(fmap.x,d1*d2*d3,1);
                          mat_tmp = mat_tmp' ;
                          [d1,d2] = size(mat_tmp);  
                          % Saving image map to testing instance matrix %
                          testing_instance_matrix(count_test,1:d2) = mat_tmp(1:d2);                       
                   end            
               end
            end
        end
    end
    count_train
    training_label_vector
    count_test
    testing_label_vector
    
%% Training the model using linear kernel%%
% Getting better accuracy with linear kernel %
model = svmtrain(double(training_label_vector), double(training_instance_matrix),'-t 0' );

%% Prediction on Training Images %%
[predicted_label_train,accuracy_train, decision_values_train] = svmpredict(double(training_label_vector), double(training_instance_matrix), model);

%% Prediction on Test Images %%
[predicted_label_test,accuracy_test, decision_values_test] = svmpredict(double(testing_label_vector), double(testing_instance_matrix), model);

accuracy_train
accuracy_test

