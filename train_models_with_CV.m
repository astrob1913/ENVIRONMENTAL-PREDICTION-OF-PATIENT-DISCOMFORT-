clc; 
clear;
close all;
rng('default');

%% 1. LOAD DATA 
data= readtable("D:\ML_Coursework\MATLAB_scripts\Medical_Env_Discomfort_Dataset.csv");
X= table2array(data(:,1:end-1));
Y= categorical(data.over_discomfort);

%% 2. TRAIN-TEST SPLIT
cvHold= cvpartition(Y,'HoldOut',0.3);
trainIdx= training(cvHold);
testIdx= test(cvHold);

Xtrain= X(trainIdx,:);
Ytrain= Y(trainIdx);

Xtest= X(testIdx,:);
Ytest= Y(testIdx);
fprintf("Train samples: %d | Test samples: %d\n",sum(trainIdx),sum(testIdx));

save('test_set.mat', 'Xtest', 'Ytest');

%% 3. Frequency-based CLASS WEIGHTS (IMBALANCE)
classCounts= countcats(Ytrain);
minorityClass= '0';
majorityClass= '1';

w0= classCounts(2)/classCounts(1);   % multiply minority
w1= 1;

freqWeights= zeros(size(Ytrain));
freqWeights(Ytrain==minorityClass) = w0;
freqWeights(Ytrain==majorityClass) = w1;

%% 4. LOGISTIC REGRESSION GRID SEARCH 
fprintf("\n   Logistic Regression Grid Search (10-fold CV)   \n");

LambdaGrid= [1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
k= 10;   % 10-fold CV
cv= cvpartition(Ytrain,"KFold",k);

LR_results= [];
for L= LambdaGrid
    f1_scores= zeros(k,1);
    for fold= 1:k
        tr= training(cv, fold);
        vl= test(cv, fold);
        mdl = fitclinear(Xtrain(tr,:), Ytrain(tr), ...
            'Learner','logistic', ...
            'Regularization','ridge', ...
            'Lambda',L, ...
            'Weights',freqWeights(tr));

        pred= categorical(predict(mdl,Xtrain(vl,:)));
        [~,~,f1]= prf1(Ytrain(vl),pred,'1');
        f1_scores(fold)= f1;
    end
    
    LR_results= [LR_results; 
    L, mean(f1_scores)];
    fprintf("Lambda = %.5f | CV-F1 = %.3f\n",L,mean(f1_scores));
end

LR_results_table= array2table(LR_results, ...
    'VariableNames',{'Lambda','CV_F1'});
LR_results_table= sortrows(LR_results_table,'CV_F1','descend');
bestLambda= LR_results_table.Lambda(1);
fprintf("\nBest Lambda= %.5f\n",bestLambda);

%% Train Final Logistic Regression
LR_best= fitclinear(Xtrain, Ytrain, ...
    'Learner','logistic', ...
    'Regularization','ridge', ...
    'Lambda',bestLambda, ...
    'Weights',freqWeights);

save("LR_best.mat","LR_best");

%% 5. RANDOM FOREST GRID SEARCH 
fprintf("\n   Random Forest Grid Search (10-fold CV)   \n");

NumTreesGrid= [10 20 50 100 150];
MinLeafGrid= [1 3 5 10];

RF_results= [];

for nt= NumTreesGrid
    for ml= MinLeafGrid
        f1_scores= zeros(k,1);
        for fold = 1:k
            tr= training(cv, fold);
            vl = test(cv, fold);

            RF= TreeBagger(nt, Xtrain(tr,:), Ytrain(tr), ...
                'Method','classification', ...
                'MinLeafSize',ml, ...
                'Weights',freqWeights(tr), ...
                'OOBPrediction','off');

            pred = categorical(predict(RF,Xtrain(vl,:)));
            [~,~,f1]= prf1(Ytrain(vl),pred,'1');
            f1_scores(fold)= f1;
        end
        
        RF_results= [RF_results;nt,ml,mean(f1_scores)];
        fprintf("Trees=%d | Leaf=%d | CV-F1=%.3f\n",nt,ml,mean(f1_scores));
    end
end

RF_results_table= array2table(RF_results, ...
    'VariableNames',{'NumTrees','MinLeaf','CV_F1'});
RF_results_table= sortrows(RF_results_table,'CV_F1','descend');
bestTrees= RF_results_table.NumTrees(1);
bestLeaf= RF_results_table.MinLeaf(1);

fprintf("\nBest RF Hyperparameters: Trees=%d | Leaf=%d\n",bestTrees,bestLeaf);

%% Train Final Random Forest
RF_best= TreeBagger(bestTrees,Xtrain,Ytrain, ...
    'Method','classification', ...
    'MinLeafSize',bestLeaf, ...
    'Weights',freqWeights, ...
    'OOBPrediction','off');

save("RF_best.mat","RF_best");

%% 6. SAVE DATASETS 
save("train_data.mat","Xtrain","Ytrain");
save("test_data.mat","Xtest","Ytest");

disp("Training complete.");

%% 10-FOLD CV for FINAL LOGISTIC REGRESSION
fprintf("\n   Running 10-fold CV for Final Logistic Regression   \n");

cv= cvpartition(Ytrain,'KFold',10);
LR_prec_scores= zeros(10,1);
LR_rec_scores= zeros(10,1);
LR_f1_scores= zeros(10,1);

for i= 1:10
    tr= training(cv,i);
    vl= test(cv,i);
    mdl = fitclinear(Xtrain(tr,:),Ytrain(tr),...
        'Learner','logistic',...
        'Regularization','ridge',...
        'Lambda',bestLambda,...
        'Weights',freqWeights(tr));

    pred= categorical(predict(mdl,Xtrain(vl,:)));
    [p, r, f]= prf1(Ytrain(vl),pred, '1');
    LR_prec_scores(i) = p;
    LR_rec_scores(i)  = r;
    LR_f1_scores(i)   = f;
end

LR_CV_Prec= mean(LR_prec_scores);
LR_CV_Rec= mean(LR_rec_scores);
LR_CV_F1= mean(LR_f1_scores);

%  Logistic Regression Training Time 
tic;
LR_best= fitclinear(Xtrain,Ytrain,...
    'Learner','logistic','Regularization','ridge',...
    'Lambda', bestLambda, 'Weights', freqWeights);
LR_Train_Time = toc;

%% 10-FOLD CV for FINAL RANDOM FOREST 
fprintf("\n   Running 10-fold CV for Final Random Forest   \n");

RF_prec_scores= zeros(10,1);
RF_rec_scores = zeros(10,1);
RF_f1_scores= zeros(10,1);

for i= 1:10
    tr= training(cv,i);
    vl= test(cv,i);
    RF_fold = TreeBagger(bestTrees, Xtrain(tr,:), Ytrain(tr), ...
        'Method','classification','MinLeafSize', bestLeaf, ...
        'Weights', freqWeights(tr), 'OOBPrediction','off');
    
    pred= categorical(predict(RF_fold, Xtrain(vl,:)));
    [p,r,f]= prf1(Ytrain(vl),pred, '1');
    RF_prec_scores(i)= p;
    RF_rec_scores(i)= r;
    RF_f1_scores(i)= f;
end

RF_CV_Prec= mean(RF_prec_scores);
RF_CV_Rec= mean(RF_rec_scores);
RF_CV_F1= mean(RF_f1_scores);

%    Random Forest Training Time 
tic;
RF_best= TreeBagger(bestTrees, Xtrain, Ytrain, ...
    'Method','classification', 'MinLeafSize',bestLeaf, ...
    'Weights',freqWeights,'OOBPrediction','off');
RF_Train_Time = toc;

%% AVG TRAIN AUC (10-FOLD CV) 
cv= cvpartition(Ytrain,'KFold',10);
LR_auc_folds= zeros(cv.NumTestSets,1);
RF_auc_folds= zeros(cv.NumTestSets,1);

for i = 1:cv.NumTestSets
    tr = training(cv,i);
    vl = test(cv,i);

    % Logistic Regression
    LR_fold = fitclinear(Xtrain(tr,:), Ytrain(tr), ...
        'Learner','logistic','Regularization','ridge', ...
        'Lambda', bestLambda,'Weights', freqWeights(tr));
    [~, scoresLR]= predict(LR_fold,Xtrain(vl,:));
    LR_probs= scoresLR(:,2);
    [~,~,~,LR_auc_folds(i)]= perfcurve(Ytrain(vl), LR_probs, '1');

    % Random Forest
    RF_fold= TreeBagger(bestTrees, Xtrain(tr,:), Ytrain(tr), ...
        'Method','classification','MinLeafSize', bestLeaf, ...
        'Weights', freqWeights(tr),'OOBPrediction','off');
    [~,scoresRF]= predict(RF_fold,Xtrain(vl,:));
    RF_probs= scoresRF(:,2);
    [~,~,~,RF_auc_folds(i)]= perfcurve(Ytrain(vl), RF_probs,'1');
end

LR_Train_AUC = mean(LR_auc_folds);
RF_Train_AUC = mean(RF_auc_folds);
fprintf("Avg Train AUC = LR: %.3f | RF: %.3f\n", LR_Train_AUC, RF_Train_AUC);

%% AVG TRAIN ERROR (10-FOLD CV) 
LR_err= zeros(cv.NumTestSets,1);
RF_err= zeros(cv.NumTestSets,1);

for i= 1:cv.NumTestSets
    tr= training(cv,i);
    vl= test(cv,i);

    % Logistic Regression
    LR_fold= fitclinear(Xtrain(tr,:), Ytrain(tr), ...
        'Learner','logistic','Lambda',bestLambda, ...
        'Weights',freqWeights(tr));
    predLR= categorical(predict(LR_fold,Xtrain(vl,:)));
    LR_err(i)= mean(predLR ~= Ytrain(vl));

    % Random Forest
    RF_fold = TreeBagger(bestTrees, Xtrain(tr,:), Ytrain(tr), ...
        'Method','classification','MinLeafSize',bestLeaf, ...
        'Weights', freqWeights(tr));
    predRF= categorical(predict(RF_fold,Xtrain(vl,:)));
    RF_err(i)= mean(predRF ~= Ytrain(vl));
end

LR_Train_Error= mean(LR_err);
RF_Train_Error= mean(RF_err);
fprintf("Avg Train Error = LR: %.3f | RF: %.3f\n", LR_Train_Error, RF_Train_Error);

%% FINAL RESULTS SUMMARY 
ResultsTable= table( ...
    ["Logistic Regression"; "Random Forest"], ...
    [LR_Train_AUC; RF_Train_AUC], ...
    [LR_Train_Error; RF_Train_Error], ...
    [LR_Train_Time; RF_Train_Time], ...                 
    'VariableNames', ...
    {'Model','AvgTrainAUC','AvgTrainError','Training_Time_sec'} );

disp("      FINAL RESULTS SUMMARY      ");
disp(ResultsTable);

%% METRIC FUNCTION 
function [precision, recall, f1] = prf1(trueLabels, predictedLabels, positiveClass)
    trueLabels = categorical(trueLabels);
    predictedLabels = categorical(predictedLabels);
    TP = sum(predictedLabels==positiveClass & trueLabels==positiveClass);
    FP = sum(predictedLabels==positiveClass & trueLabels~=positiveClass);
    FN = sum(predictedLabels~=positiveClass & trueLabels==positiveClass);
    precision = TP/(TP+FP+eps);
    recall = TP/(TP+FN+eps);
    f1 = 2*precision*recall/(precision+recall+eps);
end

save("CV_metrics.mat","LR_CV_Prec", "LR_CV_Rec", "LR_CV_F1", ...
    "RF_CV_Prec", "RF_CV_Rec", "RF_CV_F1");
