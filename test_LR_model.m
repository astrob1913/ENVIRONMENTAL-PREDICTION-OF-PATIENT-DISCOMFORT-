clc; 
clear; 
close all;

%% 1. LOAD MODEL & DATA
load("LR_best.mat");   % trained Logistic Regression model
load("test_data.mat");   % Xtest, Ytest

%% 2. PREDICTION 
LR_pred= categorical(predict(LR_best,Xtest));
[LR_Test_Prec,LR_Test_Rec,LR_Test_F1]= prf1(Ytest,LR_pred,'1');

%% 3. TEST METRICS 
LR_Test_Error= mean(LR_pred ~= Ytest);
fprintf("\nLogistic Regression — Test Results\n");
fprintf("Precision = %.3f\n",LR_Test_Prec);
fprintf("Recall    = %.3f\n",LR_Test_Rec);
fprintf("F1-score  = %.3f\n",LR_Test_F1);
fprintf("Test Error = %.3f\n",LR_Test_Error);

%% 4. CONFUSION MATRIX
figure;
confusionchart(Ytest,LR_pred);
title("Logistic Regression Confusion Matrix (Test Set)");

%% 5. ROC & AUC
[~, LR_scores]= predict( LR_best, Xtest);
LR_probs= LR_scores(:,2);   % probability of class '1'
[LR_X, LR_Y, ~,LR_AUC]= perfcurve(Ytest,LR_probs, '1');

figure;
plot(LR_X,LR_Y,'LineWidth',2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Logistic Regression ROC Curve (AUC = %.3f)', LR_AUC));
grid on;

%% 6. PREDICTION TIME 
tic;
predict(LR_best, Xtest);
LR_Predict_Time =toc;

fprintf("Prediction Time(s): %.4f\n", LR_Predict_Time);

%% METRIC FUNCTION 
function [precision, recall, f1] = prf1(trueLabels, predictedLabels, positiveClass)
    trueLabels = categorical(trueLabels);
    predictedLabels = categorical(predictedLabels);

    TP = sum(predictedLabels == positiveClass & trueLabels == positiveClass);
    FP = sum(predictedLabels == positiveClass & trueLabels ~= positiveClass);
    FN = sum(predictedLabels ~= positiveClass & trueLabels == positiveClass);

    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
end

save("LR_test_metrics.mat","LR_Test_Prec","LR_Test_Rec","LR_Test_F1");
