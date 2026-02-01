clc; 
clear; 
close all;

%%  1. LOAD MODEL & DATA 
load("RF_best.mat");   % trained Random Forest model
load("test_data.mat");    % Xtest, Ytest

%%  2. PREDICTION 
RF_pred= categorical(predict(RF_best,Xtest));
[RF_Test_Prec, RF_Test_Rec, RF_Test_F1]= prf1(Ytest,RF_pred,'1');

%%  3. TEST METRICS 
RF_Test_Error= mean(RF_pred ~= Ytest);
fprintf("\nRandom Forest — Test Results\n");
fprintf("Precision = %.3f\n",RF_Test_Prec);
fprintf("Recall    = %.3f\n",RF_Test_Rec);
fprintf("F1-score  = %.3f\n",RF_Test_F1);
fprintf("Test Error = %.3f\n",RF_Test_Error);

%%  4. CONFUSION MATRIX 
figure;
confusionchart(Ytest, RF_pred);
title("Random Forest Confusion Matrix (Test Set)");

%%  5. ROC & AUC 
[~,RF_scores]= predict(RF_best, Xtest);
RF_probs= RF_scores(:,2);   % probability of class '1'

[RF_X, RF_Y, ~, RF_AUC]= perfcurve(Ytest, RF_probs, '1');

figure;
plot(RF_X, RF_Y,'LineWidth',2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Random Forest ROC Curve (AUC = %.3f)',RF_AUC));
grid on;

%%  6. PREDICTION TIME 
tic;
predict(RF_best, Xtest);
RF_Predict_Time= toc;
fprintf("Prediction Time (s): %.4f\n",RF_Predict_Time);

%%  METRIC FUNCTION
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

save("RF_test_metrics.mat", "RF_Test_Prec", "RF_Test_Rec", "RF_Test_F1");
