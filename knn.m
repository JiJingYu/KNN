%% format

clc
clear
%% load data

load('mnist_test_data.mat');
test_data = images;
load('mnist_train_data.mat');
train_data = images;
load('mnist_test_label.mat');
test_labels = labels;
load('mnist_train_label.mat');
train_labels = labels;

test_data = reshape(test_data, size(test_data, 1) * size(test_data, 2), []);
train_data = reshape(train_data, size(train_data, 1) * size(train_data, 2), []);
%% for train_data, each classification 100 points

train_data_2 = [];
train_label_2 = [];
class_all = sort(unique( train_labels));
for iCount = 1:size( class_all, 1)
    curr_label = class_all( iCount);
    curr_lndex = find( train_labels == curr_label );
    data_buf = train_data( :, curr_lndex( 1:300 ));
    label_buf = train_labels( curr_lndex( 1:300 ));
    train_data_2 = cat( 2, train_data_2, data_buf);
    train_label_2 = cat( 1, train_label_2, label_buf);
end

%% for each test_data, compute loss function and 
pred_label = zeros(size(test_labels));
for iCount = 1:size( test_data, 2)
    curr_test_data = test_data(:, iCount);
    test_data_buf = repmat( curr_test_data, 1, size( train_data_2, 2));
    loss = sum(abs(train_data_2 - test_data_buf));
    [~, index] = sort(loss);
    pred_label(iCount) = mode( train_label_2( index(1: 7)));
end